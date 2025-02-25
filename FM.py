import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBankV2(nn.Module):
    def __init__(self, class_count, feature_dim, transform_dim, output_dim, enable_hard_clustering=False,
                 downscale_before_attention=False, normalization_config=None, activation_config=None,
                 corner_alignment=False):
        super(MemoryBankV2, self).__init__()
        self.corner_alignment = corner_alignment
        self.class_count = class_count
        self.feature_dim = feature_dim
        self.transform_dim = transform_dim
        self.output_dim = output_dim
        self.enable_hard_clustering = enable_hard_clustering

        # Downscale before applying attention if enabled
        if downscale_before_attention:
            self.downscale_before_attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
                NormalizationLayer(placeholder=feature_dim, norm_cfg=normalization_config),
                ActivationLayer(activation_config),
            )

        # Initialize memory with random normal distribution for better initialization
        self.memory_bank = nn.Parameter(torch.randn(class_count, 1, feature_dim), requires_grad=False)

        # Define the attention module
        self.attention_module = AttentionModule(
            key_channels=feature_dim, query_channels=feature_dim, transform_channels=transform_dim,
            output_channels=feature_dim,
            share_key_query=False, query_downscale=None, key_downscale=None, key_query_conv_count=2,
            value_output_conv_count=1, key_query_normalize=True,
            value_output_normalize=True, matrix_normalization=True, with_output_projection=True,
            norm_cfg=normalization_config, act_cfg=activation_config,
        )

        # Fusion bottleneck with residual connection
        self.fusion_bottleneck = nn.Sequential(
            nn.Conv2d(feature_dim * 2, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            NormalizationLayer(placeholder=output_dim, norm_cfg=normalization_config),
            ActivationLayer(activation_config),
        )

    def forward(self, input_features, prediction_map=None):
        batch_size, channel_count, height, width = input_features.size()

        # Compute class-specific weights (using softmax over predictions)
        class_weights = prediction_map.permute(0, 2, 3, 1).contiguous().view(-1, self.class_count)
        class_weights = F.softmax(class_weights, dim=-1)

        # Optionally apply hard clustering (one-hot encoding)
        if self.enable_hard_clustering:
            class_labels = class_weights.argmax(dim=-1).unsqueeze(-1)
            one_hot_labels = torch.zeros_like(class_weights).scatter_(1, class_labels.long(), 1)
            class_weights = one_hot_labels

        # Initialize memory representation from the memory bank
        memory_content = self.memory_bank.data.clone()  # (class_count, 1, feature_dim)

        # Aggregate memory using class weights
        aggregated_memory = torch.matmul(class_weights,
                                         memory_content.view(self.class_count, -1))  # (B*H*W, feature_dim)

        # Reshape aggregated memory to match spatial dimensions
        aggregated_memory = aggregated_memory.view(batch_size, height, width, channel_count).permute(0, 3, 1, 2)

        # Optionally downscale input features and aggregated memory before applying attention
        if hasattr(self, 'downscale_before_attention'):
            feature_input, aggregated_memory_input = self.downscale_before_attention(
                input_features), self.downscale_before_attention(aggregated_memory)
        else:
            feature_input, aggregated_memory_input = input_features, aggregated_memory

        # Apply attention module
        aggregated_memory = self.attention_module(feature_input, aggregated_memory_input)

        # Upscale if downscale was applied before attention
        if hasattr(self, 'downscale_before_attention'):
            aggregated_memory = F.interpolate(aggregated_memory, size=input_features.size()[2:], mode='bilinear',
                                              align_corners=self.corner_alignment)

        # Concatenate features and aggregated memory, then pass through the fusion bottleneck
        output_memory = self.fusion_bottleneck(torch.cat([input_features, aggregated_memory], dim=1))

        return self.memory_bank.data, output_memory

    def update_memory(self, input_features, segmentation_map, ignore_value=255, momentum_config=None,
                      learning_rate=None):
        batch_size, feature_count, height, width = input_features.size()
        momentum_factor = momentum_config['base_momentum']

        if momentum_config['adjust_by_learning_rate']:
            momentum_factor = momentum_config['base_momentum'] / momentum_config['base_lr'] * learning_rate

        # Flatten features for easier processing
        input_features = input_features.permute(0, 2, 3, 1).contiguous().view(batch_size * height * width,
                                                                              feature_count)
        segmentation_map = segmentation_map.long()

        # Update memory based on segmentation
        unique_class_ids = segmentation_map.unique()
        for class_id in unique_class_ids:
            if class_id == ignore_value:
                continue
            # Select features corresponding to the current class
            segmentation_class = segmentation_map.view(-1)
            class_features = input_features[segmentation_class == class_id]

            # Update memory with the mean and std of the selected features
            class_mean = class_features.mean(0)
            class_std = class_features.std(0)
            self.memory_bank[class_id, 0] = (1 - momentum_factor) * self.memory_bank[
                class_id, 0] + momentum_factor * class_mean
            self.memory_bank[class_id, 1] = (1 - momentum_factor) * self.memory_bank[
                class_id, 1] + momentum_factor * class_std

        # Return updated memory content
        return self.memory_bank.data
