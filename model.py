import torch
from torch import nn


class ConditionalBatchNorm1d(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.module_dict = nn.ModuleDict(dict(
            batch_norm1d=nn.BatchNorm1d(out_channels, affine=False),
            conv1d_gamma=nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True),
            conv1d_beta=nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        ))

    def forward(self, inputs, conditions):
        inputs = self.module_dict.batch_norm1d(inputs)
        gamma = self.module_dict.conv1d_gamma(conditions)
        beta = self.module_dict.conv1d_beta(conditions)
        inputs = inputs * gamma + beta
        return inputs


class ConditionalResidual1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, condition_channels, projection=False):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            first_conv_block=nn.ModuleDict(dict(
                conditional_batch_norm1d=ConditionalBatchNorm1d(condition_channels, in_channels),
                relu=nn.ReLU(),
                conv1d=nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )),
            second_linear_block=nn.ModuleDict(dict(
                conditional_batch_norm1d=ConditionalBatchNorm1d(condition_channels, out_channels),
                relu=nn.ReLU(),
                conv1d=nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
            )),
            projection_block=nn.ModuleDict(dict(
                conv1d=nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )) if projection else None
        ))

    def forward(self, inputs, conditions):

        shortcut = inputs

        inputs = self.module_dict.first_linear_block.conditional_batch_norm1d(inputs, conditions)
        inputs = self.module_dict.first_linear_block.relu(inputs)

        if self.module_dict.projection_block:
            shortcut = self.module_dict.projection_block.conv1d(inputs)

        inputs = self.module_dict.first_linear_block.conv1d(inputs)

        inputs = self.module_dict.second_linear_block.conditional_batch_norm1d(inputs, conditions)
        inputs = self.module_dict.second_linear_block.relu(inputs)
        inputs = self.module_dict.second_linear_block.conv1d(inputs)

        inputs = inputs + shortcut

        return inputs


class ConditionalResidualDecoder(nn.Module):

    def __init__(self, position_conv_param, latent_conv_param, conditional_residual_params, conv_param):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            position_conv1d_block=nn.Sequential(
                nn.Conv1d(**position_conv_param)
            ),
            latent_conv1d_block=nn.Sequential(
                nn.Conv1d(**latent_conv_param)
            ),
            conditional_residual1d_blocks=nn.ModuleList([
                ConditionalResidual1dBlock(**conditional_residual_param)
                for conditional_residual_param in conditional_residual_params
            ]),
            conv1d_block=nn.Sequential(
                ConditionalBatchNorm1d(
                    conditional_residual_params[-1].condition_channels,
                    conditional_residual_params[-1].out_channels
                ),
                nn.ReLU(),
                nn.Conv1d(**conv_param)
            )
        ))

    def forward(self, positions, latents, conditions):

        inputs = self.module_dict.position_conv1d_block(positions)

        if latents is not None:
            inputs = inputs + self.module_dict.latent_conv1d_block(latents)

        for conditional_residual1d_block in self.module_dict.conditional_residual1d_blocks:
            inputs = conditional_residual1d_block(inputs, conditions)

        inputs = self.module_dict.conv1d_block(inputs)

        return inputs
