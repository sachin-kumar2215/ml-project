import torch
import torch.nn as nn
from typing import List


class CNNSentimentClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) for sentiment classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: int,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.3,
    ):
        super(CNNSentimentClassifier, self).__init__()

        # Embedding layer: Converts word indices to dense vectors
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )  # padding_idx=0 for <PAD> token

        # Convolutional layers with different filter sizes
        # nn.ModuleList is used to hold multiple nn.Module objects
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=size
                )
                for size in filter_sizes
            ]
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected (linear) layer for classification
        # Input features to FC layer = (number of filter sizes) * (number of filters per size)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len],
                              containing word indices.
            lengths (torch.Tensor, optional): Tensor of actual sequence lengths.
                                              Not directly used by this CNN architecture
                                              but kept for compatibility with DataLoader.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim],
                          containing raw scores for each class.
        """
        # x: [batch_size, seq_len]

        # Embedding: Convert word indices to dense vectors
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # Transpose for Conv1d: Conv1d expects input channels as the second dimension.
        # So, we change from [batch_size, seq_len, embed_dim] to [batch_size, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)  # Equivalent to .transpose(1, 2)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU activation
            conv_out = torch.relu(
                conv(embedded)
            )  # [batch_size, num_filters, conv_seq_len]

            # Apply max pooling over the sequence dimension (dim=2)
            # This extracts the most important feature (max value) from each filter's output.
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)

        # Concatenate all max-pooled outputs from different filter sizes
        # This creates a single feature vector for each sample in the batch.
        concatenated = torch.cat(
            conv_outputs, dim=1
        )  # [batch_size, len(filter_sizes) * num_filters]

        # Apply dropout for regularization
        concatenated = self.dropout(concatenated)

        # Final classification layer
        output = self.fc(concatenated)  # [batch_size, output_dim]

        return output
