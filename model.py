"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
import torch

def edges_to_coo(edge_list, num_nodes=None):
    """
    Convert graph edges into COO index tensors.
    
    Args:
        edge_list: Python list of (src, dst) integer pairs or LongTensor of shape [E, 2]
        num_nodes: Optional number of nodes. If None, infer from edges.
    
    Returns:
        src: 1-D LongTensor of source indices
        dst: 1-D LongTensor of destination indices
        num_nodes: Number of nodes
    """
    # Handle LongTensor input
    if torch.is_tensor(edge_list):
        # Assume shape [E, 2]
        src = edge_list[:, 0]
        dst = edge_list[:, 1]
        edges = edge_list
    else:
        # Handle Python list input
        if len(edge_list) == 0:
            src = torch.tensor([], dtype=torch.long)
            dst = torch.tensor([], dtype=torch.long)
            edges = []
        else:
            # Convert list of tuples to tensors
            src_list = [e[0] for e in edge_list]
            dst_list = [e[1] for e in edge_list]
            src = torch.tensor(src_list, dtype=torch.long)
            dst = torch.tensor(dst_list, dtype=torch.long)
            edges = edge_list
    
    # Determine number of nodes
    if num_nodes is None:
        if len(edges) == 0:
            num_nodes = 0
        else:
            # Find maximum node index
            if torch.is_tensor(edges):
                max_idx = torch.max(edges).item()
            else:
                max_idx = max(max(e[0], e[1]) for e in edges)
            num_nodes = max_idx + 1
    
    return src, dst, num_nodes

# Step 2 - add_self_loops
import torch

def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # Create self-loop indices for all nodes
    self_loops = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    
    # Concatenate original edges with self-loops
    src_out = torch.cat([src, self_loops])
    dst_out = torch.cat([dst, self_loops])
    
    return src_out, dst_out

# Step 3 - compute_node_degrees
import torch

def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # If edge_weight is not provided, use ones
    if edge_weight is None:
        # Create tensor of ones with float dtype
        values = torch.ones(dst.size(0), dtype=torch.float, device=dst.device)
    else:
        # Ensure edge_weight is float
        values = edge_weight.to(torch.float)
    
    # Scatter values to destination nodes using index_add_
    # Initialize degree tensor with zeros
    degrees = torch.zeros(num_nodes, dtype=torch.float, device=dst.device)
    
    # Add values to the corresponding destination nodes
    degrees.index_add_(0, dst, values)
    
    return degrees

# Step 4 - symmetric_normalize_edge_weights
import torch

def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # If edge_weight is None, use ones
    if edge_weight is None:
        edge_weight = torch.ones(src.size(0), dtype=torch.float32, device=src.device)
    else:
        # Ensure edge_weight is float32
        edge_weight = edge_weight.to(torch.float32)
    
    # Compute (weighted) in-degrees for each node
    deg = compute_node_degrees(src, dst, num_nodes, edge_weight)
    
    # Compute inverse square root of degrees, treating 0 as 0
    # For degree > 0, compute 1/sqrt(deg), otherwise 0
    inv_sqrt_deg = torch.zeros_like(deg)
    nonzero_mask = deg > 0
    inv_sqrt_deg[nonzero_mask] = 1.0 / torch.sqrt(deg[nonzero_mask])
    
    # Get inverse square root for source and destination nodes
    inv_sqrt_src = inv_sqrt_deg[src]
    inv_sqrt_dst = inv_sqrt_deg[dst]
    
    # Normalize: w_ij / sqrt(d_i * d_j) = w_ij * inv_sqrt(d_i) * inv_sqrt(d_j)
    normalized_weights = edge_weight * inv_sqrt_src * inv_sqrt_dst
    
    return normalized_weights

# Step 5 - gather_source_node_features
import torch

def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.
    # Use standard tensor indexing to gather features for source nodes
    return node_features[src]

# Step 6 - scatter_sum_to_nodes
import torch

def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # Get the feature dimension and device/dtype from edge_features
    E, F = edge_features.shape
    device = edge_features.device
    dtype = edge_features.dtype
    
    # Initialize output tensor with zeros
    node_features = torch.zeros(num_nodes, F, dtype=dtype, device=device)
    
    # Scatter-sum edge features to destination nodes using index_add_
    # For each edge i with destination dst[i], add edge_features[i] to node_features[dst[i]]
    node_features.index_add_(0, dst, edge_features)
    
    return node_features

# Step 7 - scatter_mean_to_nodes
import torch

def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    # First, compute the sum of edge features for each node
    sums = scatter_sum_to_nodes(edge_features, dst, num_nodes)
    
    # Then compute the count of edges for each node
    # Use ones with the same device as edge_features
    ones = torch.ones(edge_features.size(0), dtype=edge_features.dtype, device=edge_features.device)
    counts = scatter_sum_to_nodes(ones.unsqueeze(-1), dst, num_nodes)  # Shape: (num_nodes, 1)
    
    # Avoid division by zero: where count is 0, keep the sum as 0 (mean will be 0)
    # For nodes with count > 0, divide sum by count
    # Use broadcasting to handle the division
    counts_squeezed = counts.squeeze(-1)  # Shape: (num_nodes,)
    
    # Create mask for nodes with at least one incoming edge
    mask = counts_squeezed > 0
    
    # Initialize result with zeros
    result = torch.zeros_like(sums)
    
    # For nodes with incoming edges, compute mean
    # Need to handle broadcasting carefully: sums[mask] is (M, F), counts[mask] is (M,)
    result[mask] = sums[mask] / counts_squeezed[mask].unsqueeze(-1)
    
    return result

# Step 8 - scatter_max_to_nodes
import torch

def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    E, F = edge_features.shape
    device = edge_features.device
    dtype = edge_features.dtype
    
    # Initialize with -inf for all nodes and features
    result = torch.full((num_nodes, F), float('-inf'), dtype=dtype, device=device)
    
    # Use index_reduce_ with 'amax' operation to compute elementwise max
    # This is more efficient than looping
    result.index_reduce_(0, dst, edge_features, reduce='amax', include_self=False)
    
    return result

# Step 9 - compute_messages
import torch

def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # Gather source and destination node features
    src_features = gather_source_node_features(node_features, src)
    dst_features = gather_source_node_features(node_features, dst)
    
    # Apply message_fn with appropriate arguments
    if edge_attr is not None:
        messages = message_fn(src_features, dst_features, edge_attr)
    else:
        messages = message_fn(src_features, dst_features)
    
    return messages

# Step 10 - aggregate_messages
import torch

def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    if aggr == 'sum':
        return scatter_sum_to_nodes(messages, dst, num_nodes)
    elif aggr == 'mean':
        return scatter_mean_to_nodes(messages, dst, num_nodes)
    elif aggr == 'max':
        return scatter_max_to_nodes(messages, dst, num_nodes)
    else:
        raise ValueError(f"Unknown aggregation mode: {aggr}. Must be 'sum', 'mean', or 'max'.")

# Step 11 - update_node_features
import torch

def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    # Apply the update function to fuse node features with aggregated messages
    return update_fn(node_features, aggregated)

# Step 12 - message_passing_layer
import torch

def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # Step 1: Compute messages
    messages = compute_messages(node_features, src, dst, message_fn, edge_attr)
    
    # Step 2: Aggregate messages onto destination nodes
    num_nodes = node_features.shape[0]
    aggregated = aggregate_messages(messages, dst, num_nodes, aggr)
    
    # Step 3: Update node features
    updated_features = update_node_features(node_features, aggregated, update_fn)
    
    return updated_features

# Step 13 - stack_message_passing_layers
import torch

def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # If no layers, return input unchanged and empty list
    if not layers:
        return node_features, []
    
    # Initialize with input features
    current_features = node_features
    all_layer_outputs = []
    
    # Apply each layer sequentially
    for layer in layers:
        current_features = layer(current_features, src, dst, edge_attr)
        all_layer_outputs.append(current_features)
    
    # Final embeddings is the output of the last layer
    embeddings = current_features
    
    return embeddings, all_layer_outputs

# Step 14 - gcn_renormalize_adjacency
import torch

def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # Step 1: Add self-loops
    src_hat, dst_hat = add_self_loops(src, dst, num_nodes)
    
    # Step 2: All edge weights are 1.0 (unweighted graph)
    # Create ones for all edges (original + self-loops)
    E_hat = src_hat.size(0)
    edge_weight = torch.ones(E_hat, dtype=torch.float32, device=src.device)
    
    # Step 3: Symmetrically normalize the augmented adjacency
    norm_weight = symmetric_normalize_edge_weights(src_hat, dst_hat, num_nodes, edge_weight)
    
    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
import torch

def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # Compute matrix product: node_features @ weight
    transformed = node_features @ weight
    
    # Add bias if provided
    if bias is not None:
        transformed = transformed + bias
    
    return transformed

# Step 16 - gcn_layer_forward
import torch

def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # Set num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    # Step 1: GCN renormalization - add self-loops and compute normalized weights
    src_hat, dst_hat, norm_weights = gcn_renormalize_adjacency(src, dst, num_nodes)
    
    # Step 2: Apply linear transform to node features
    transformed_features = gcn_linear_transform(node_features, weight, bias)
    
    # Step 3: Message passing with GCN propagation
    # Messages are the transformed features of the source nodes
    # We need to gather transformed features for source nodes and weight them
    gathered_src = gather_source_node_features(transformed_features, src_hat)
    
    # Apply normalized weights to messages (elementwise multiplication)
    # norm_weights has shape (E_hat,), gathered_src has shape (E_hat, Fout)
    # We need to broadcast norm_weights to multiply each feature dimension
    weighted_messages = gathered_src * norm_weights.unsqueeze(-1)
    
    # Aggregate messages to destination nodes using sum
    aggregated = scatter_sum_to_nodes(weighted_messages, dst_hat, num_nodes)
    
    # Step 4: Apply activation if provided
    if activation is not None:
        aggregated = activation(aggregated)
    
    return aggregated

# Step 17 - init_gcn_parameters
import torch

def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Calculate Glorot uniform bound: sqrt(6 / (in_dim + out_dim))
    a = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
    
    # Initialize weight matrix with uniform distribution in [-a, a]
    weight = torch.empty(in_dim, out_dim).uniform_(-a, a)
    
    # Create parameter dictionary
    params = {'weight': weight}
    
    # Add bias if requested
    if with_bias:
        bias = torch.zeros(out_dim)
        params['bias'] = bias
    
    return params

# Step 18 - gcn_stack_forward
import torch

def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # Set num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    # If activations is None, create a list of None for each layer
    if activations is None:
        activations = [None] * len(param_list)
    
    # Ensure activations list matches param_list length
    if len(activations) != len(param_list):
        raise ValueError(f"activations length ({len(activations)}) must match param_list length ({len(param_list)})")
    
    # Initialize with input features
    current_features = node_features
    all_layer_outputs = []
    
    # Apply each GCN layer sequentially
    for params, activation in zip(param_list, activations):
        # Extract weight and optional bias from params
        weight = params['weight']
        bias = params.get('bias', None)
        
        # Apply GCN layer
        current_features = gcn_layer_forward(
            current_features, 
            src, 
            dst, 
            weight, 
            bias=bias, 
            num_nodes=num_nodes, 
            activation=activation
        )
        
        # Store the output of this layer
        all_layer_outputs.append(current_features)
    
    # Final embeddings is the output of the last layer
    embeddings = current_features
    
    return embeddings, all_layer_outputs

# Step 19 - gat_attention_logits
import torch
import torch.nn.functional as F

def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # Step 1: Apply linear transform to all nodes
    # transformed = X @ W (no bias)
    transformed = node_features @ weight  # Shape: (N, Fout)
    
    # Step 2: Compute attention scores for each edge
    # Gather transformed features for source and destination nodes
    transformed_src = transformed[src]  # Shape: (E, Fout)
    transformed_dst = transformed[dst]  # Shape: (E, Fout)
    
    # Compute attention logits: e_ij = LeakyReLU(attn_src^T (W h_i) + attn_dst^T (W h_j))
    # attn_src and attn_dst are shape (Fout,)
    # attn_src^T (W h_i) = dot product of attn_src with transformed_src
    src_attn = torch.sum(attn_src * transformed_src, dim=-1)  # Shape: (E,)
    dst_attn = torch.sum(attn_dst * transformed_dst, dim=-1)  # Shape: (E,)
    
    # Sum and apply LeakyReLU with negative_slope=0.2
    logits = F.leaky_relu(src_attn + dst_attn, negative_slope=0.2)
    
    return logits, transformed

# Step 20 - gat_masked_neighbor_softmax
import torch

def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # Step 1: Compute max logit for each destination node
    # Initialize with -inf for all nodes
    max_per_node = torch.full((num_nodes,), float('-inf'), device=logits.device, dtype=logits.dtype)
    
    # Use index_reduce to compute max for each destination
    max_per_node.index_reduce_(0, dst, logits, reduce='amax', include_self=False)
    
    # Step 2: Subtract max for numerical stability and compute exp
    # For each edge, subtract the max of its destination node
    logits_shifted = logits - max_per_node[dst]
    exp_logits = torch.exp(logits_shifted)
    
    # Step 3: Compute sum of exp for each destination node
    # Initialize with zeros
    sum_per_node = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    
    # Scatter-sum the exponentials to destination nodes
    sum_per_node.index_add_(0, dst, exp_logits)
    
    # Step 4: Normalize to get softmax probabilities
    # For each edge, divide exp by the sum of its destination node
    coefficients = exp_logits / sum_per_node[dst]
    
    # Handle potential NaN from division by zero (if a node has no incoming edges)
    # But coefficients for such edges wouldn't exist anyway since dst only has valid entries
    coefficients = torch.nan_to_num(coefficients, nan=0.0)
    
    return coefficients

# Step 21 - gat_head_forward
import torch

def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # Set num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    # Step 1: Compute attention logits and transformed features
    logits, transformed = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)
    
    # Step 2: Apply masked softmax to get attention coefficients
    attn_coeffs = gat_masked_neighbor_softmax(logits, dst, num_nodes)
    
    # Step 3: Aggregate messages
    # Gather transformed features for source nodes
    transformed_src = transformed[src]  # Shape: (E, Fout)
    
    # Weight messages by attention coefficients
    weighted_messages = transformed_src * attn_coeffs.unsqueeze(-1)  # Shape: (E, Fout)
    
    # Aggregate weighted messages to destination nodes
    head_out = scatter_sum_to_nodes(weighted_messages, dst, num_nodes)  # Shape: (N, Fout)
    
    # Step 4: Add bias if provided
    if bias is not None:
        head_out = head_out + bias
    
    # Step 5: Apply activation if provided
    if activation is not None:
        head_out = activation(head_out)
    
    return head_out, attn_coeffs

# Step 22 - merge_gat_heads
import torch

def merge_gat_heads(head_outputs, mode='concat'):
    # TODO: Merge multi-head GAT outputs into one node-feature tensor.
    # Check if head_outputs is a list/tuple or a stacked tensor
    if isinstance(head_outputs, (list, tuple)):
        # Convert list of tensors to stacked tensor
        stacked = torch.stack(head_outputs, dim=0)  # Shape: (H, N, F)
    elif torch.is_tensor(head_outputs):
        # Assume it's already stacked with shape (H, N, F)
        stacked = head_outputs
    else:
        raise ValueError(f"head_outputs must be a list/tuple of tensors or a stacked tensor, got {type(head_outputs)}")
    
    # Validate mode
    if mode == 'concat':
        # Concatenate along the feature dimension
        # stacked shape: (H, N, F) -> permute to (N, H, F) -> reshape to (N, H*F)
        merged = stacked.permute(1, 0, 2).reshape(stacked.shape[1], -1)
    elif mode == 'mean':
        # Average across the head dimension
        merged = stacked.mean(dim=0)  # Shape: (N, F)
    else:
        raise ValueError(f"mode must be 'concat' or 'mean', got '{mode}'")
    
    return merged

# Step 23 - gat_layer_forward
import torch

def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    # Set num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    # Run each attention head
    head_outputs = []
    all_attn = []
    
    for params in head_params:
        # Extract parameters for this head
        weight = params['weight']
        attn_src = params['attn_src']
        attn_dst = params['attn_dst']
        bias = params.get('bias', None)
        
        # Run single head forward pass
        head_out, attn_coeffs = gat_head_forward(
            node_features, 
            src, 
            dst, 
            weight, 
            attn_src, 
            attn_dst, 
            bias=bias, 
            num_nodes=num_nodes,
            activation=None  # Activation applied after merging
        )
        
        head_outputs.append(head_out)
        all_attn.append(attn_coeffs)
    
    # Merge head outputs
    merged = merge_gat_heads(head_outputs, mode=merge_mode)
    
    # Apply activation if provided
    if activation is not None:
        merged = activation(merged)
    
    return merged, all_attn

# Step 24 - init_gat_parameters
import torch

def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    # TODO: Initialize multi-head GAT parameters with Glorot-style initialization.
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    head_params = []
    
    for _ in range(num_heads):
        # Initialize weight matrix: shape (in_dim, out_dim)
        # Glorot uniform with fan_in=in_dim, fan_out=out_dim
        a_weight = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
        weight = torch.empty(in_dim, out_dim).uniform_(-a_weight, a_weight)
        weight.requires_grad_(True)
        
        # Initialize source attention vector: shape (out_dim,)
        # Glorot uniform with fan_in=out_dim, fan_out=1
        a_attn = torch.sqrt(torch.tensor(6.0 / (out_dim + 1)))
        attn_src = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_src.requires_grad_(True)
        
        # Initialize destination attention vector: shape (out_dim,)
        # Glorot uniform with fan_in=out_dim, fan_out=1
        attn_dst = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_dst.requires_grad_(True)
        
        # Create parameter dict
        params = {
            'weight': weight,
            'attn_src': attn_src,
            'attn_dst': attn_dst
        }
        
        # Add bias if requested
        if with_bias:
            bias = torch.zeros(out_dim)
            bias.requires_grad_(True)
            params['bias'] = bias
        
        head_params.append(params)
    
    return head_params

# Step 25 - gat_stack_forward
import torch

def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # Set num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    # Set default merge_modes if not provided
    if merge_modes is None:
        merge_modes = ['concat'] * len(layer_param_list)
    
    # Set default activations if not provided
    if activations is None:
        activations = [None] * len(layer_param_list)
    
    # Validate lengths
    if len(merge_modes) != len(layer_param_list):
        raise ValueError(f"merge_modes length ({len(merge_modes)}) must match layer_param_list length ({len(layer_param_list)})")
    if len(activations) != len(layer_param_list):
        raise ValueError(f"activations length ({len(activations)}) must match layer_param_list length ({len(layer_param_list)})")
    
    # Initialize with input features
    current_features = node_features
    all_layer_outputs = []
    
    # Apply each GAT layer sequentially
    for head_params, merge_mode, activation in zip(layer_param_list, merge_modes, activations):
        # Apply GAT layer
        current_features, _ = gat_layer_forward(
            current_features,
            src,
            dst,
            head_params,
            merge_mode=merge_mode,
            num_nodes=num_nodes,
            activation=activation
        )
        
        # Store the output of this layer
        all_layer_outputs.append(current_features)
    
    # Final embeddings is the output of the last layer
    embeddings = current_features
    
    return embeddings, all_layer_outputs

# Step 26 - global_mean_pool
import torch

def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # Determine number of graphs
    if num_graphs is None:
        num_graphs = batch_index.max().item() + 1
    
    # Step 1: Sum features for each graph using scatter_sum_to_nodes
    # Treat batch_index as the "destination" for scattering node features
    summed_features = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    
    # Step 2: Count number of nodes in each graph
    # Use ones with the same device and dtype as node_features
    ones = torch.ones(node_features.size(0), dtype=node_features.dtype, device=node_features.device)
    counts = scatter_sum_to_nodes(ones.unsqueeze(-1), batch_index, num_graphs)  # Shape: (num_graphs, 1)
    counts = counts.squeeze(-1)  # Shape: (num_graphs,)
    
    # Step 3: Compute mean by dividing sum by count
    # Avoid division by zero (shouldn't happen if all graphs have at least one node)
    mean_features = summed_features / counts.unsqueeze(-1)
    
    # Handle potential NaN from division by zero (if any graph has no nodes)
    mean_features = torch.nan_to_num(mean_features, nan=0.0)
    
    return mean_features

# Step 27 - global_sum_pool
import torch

def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # Determine number of graphs
    if num_graphs is None:
        num_graphs = batch_index.max().item() + 1
    
    # Sum features for each graph using scatter_sum_to_nodes
    # Treat batch_index as the "destination" for scattering node features
    summed_features = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    
    return summed_features

# Step 28 - global_max_pool
import torch

def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    # Determine number of graphs
    if num_graphs is None:
        num_graphs = batch_index.max().item() + 1
    
    # Get feature dimension and device
    F = node_features.shape[1]
    device = node_features.device
    dtype = node_features.dtype
    
    # Initialize with -inf for all graphs and features
    max_features = torch.full((num_graphs, F), float('-inf'), dtype=dtype, device=device)
    
    # Use index_reduce_ with 'amax' to compute elementwise max
    # batch_index is used as the "destination" for reduction
    max_features.index_reduce_(0, batch_index, node_features, reduce='amax', include_self=False)
    
    # Note: -inf rows remain for graphs that have no nodes
    # For graphs with nodes, all values will be finite (since node_features are finite)
    
    return max_features

# Step 29 - global_mean_max_pool
import torch

def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # Determine number of graphs
    if num_graphs is None:
        num_graphs = batch_index.max().item() + 1
    
    # Compute mean pool
    mean_pooled = global_mean_pool(node_features, batch_index, num_graphs)
    
    # Compute max pool
    max_pooled = global_max_pool(node_features, batch_index, num_graphs)
    
    # Concatenate along the feature dimension
    combined = torch.cat([mean_pooled, max_pooled], dim=-1)
    
    return combined

# Step 30 - node_classification_head
import torch

def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
    # Compute matrix product: node_embeddings @ weight
    logits = node_embeddings @ weight
    
    # Add bias if provided
    if bias is not None:
        logits = logits + bias
    
    return logits

# Step 31 - graph_regression_head
import torch

def graph_regression_head(graph_embeddings, weight, bias=None):
    # TODO: Map pooled graph embeddings to regression predictions via a linear head.
    # Compute matrix product: graph_embeddings @ weight.T
    # weight is of shape (out_dim, D), so we need to transpose it for multiplication
    predictions = graph_embeddings @ weight.T
    
    # Add bias if provided
    if bias is not None:
        predictions = predictions + bias
    
    return predictions

# Step 32 - generate_sbm_graph
import torch

def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Sample one SBM graph with community labels and random node features.
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Step 1: Assign community labels in contiguous blocks
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start = c * num_nodes // num_classes
        end = (c + 1) * num_nodes // num_classes
        node_labels[start:end] = c
    
    # Step 2: Sample node features from standard normal distribution
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Step 3: Generate edges based on SBM probabilities
    src_list = []
    dst_list = []
    
    # Iterate over all pairs of nodes (i, j) with i < j to avoid duplicate undirected edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Determine if nodes are in the same community
            same_community = (node_labels[i] == node_labels[j])
            
            # Choose probability based on community membership
            prob = p_in if same_community else p_out
            
            # Sample edge with probability prob
            if torch.rand(1).item() < prob:
                # Add both directed edges (i -> j and j -> i)
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
    
    # Convert to tensors
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    
    # Stack to create edge_index of shape (2, E)
    edge_index = torch.stack([src, dst])
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_labels': node_labels,
        'num_nodes': num_nodes
    }

# Step 33 - build_node_classification_dataset
import torch

def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Build a list of SBM graphs with consistent schema for node classification.
    graphs = []
    
    for i in range(num_graphs):
        # Derive a distinct per-graph seed
        if seed is not None:
            # Use the base seed + i to ensure different graphs but reproducible
            graph_seed = seed + i
        else:
            graph_seed = None
        
        # Generate one SBM graph
        graph = generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=graph_seed)
        graphs.append(graph)
    
    return graphs

# Step 34 - generate_molecule_like_graph
import torch

def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob, seed=None):
    # TODO: Synthesize one molecule-like graph for graph-level regression.
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Step 1: Generate node features from standard normal
    x = torch.randn(num_nodes, num_node_features)
    
    # Step 2: Generate edges
    src_list = []
    dst_list = []
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    
    # Iterate over all pairs (i, j) with i < j
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Sample edge with probability edge_prob
            if torch.rand(1).item() < edge_prob:
                # Add both directed edges
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
                # Increment degrees for both nodes
                degrees[i] += 1
                degrees[j] += 1
    
    # Convert to tensors
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    edge_index = torch.stack([src, dst])  # Shape: (2, E)
    
    # Step 3: Compute target y
    # y = mean over nodes of deg(v) * mean(x[v])
    # mean(x[v]) is the mean of node features across feature dimension
    node_mean = x.mean(dim=-1)  # Shape: (num_nodes,)
    deg_times_mean = degrees.float() * node_mean  # Shape: (num_nodes,)
    y = deg_times_mean.mean()  # Scalar
    
    return {
        'x': x,
        'edge_index': edge_index,
        'y': y
    }

# Step 35 - build_graph_regression_dataset
import torch

def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Build a list of molecule-like graphs for graph-level regression.
    lo, hi = num_nodes_range
    
    graphs = []
    for i in range(num_graphs):
        # Calculate number of nodes for this graph
        num_nodes = lo + (i % (hi - lo + 1))
        
        # Generate graph with distinct seed
        graph = generate_molecule_like_graph(
            num_nodes, 
            num_node_features, 
            edge_prob, 
            seed=seed + i
        )
        graphs.append(graph)
    
    return graphs

# Step 36 - collate_graph_batch
import torch

def collate_graph_batch(graphs):
    # TODO: Combine variable-size graphs into one disconnected batched graph.
    num_graphs = len(graphs)
    
    # Lists to collect batched data
    batched_x = []
    batched_edge_index_src = []
    batched_edge_index_dst = []
    batched_y = []
    batched_batch = []
    
    # Cumulative node offset for shifting edge indices
    node_offset = 0
    
    for graph_id, graph in enumerate(graphs):
        # Get node features for this graph
        x = graph['x']
        edge_index = graph['edge_index']
        y = graph['y']
        
        # Number of nodes in this graph
        num_nodes = x.shape[0]
        
        # Collect node features
        batched_x.append(x)
        
        # Collect batch indices for nodes in this graph
        batched_batch.extend([graph_id] * num_nodes)
        
        # Collect target y (convert to tensor if needed)
        if isinstance(y, (int, float)):
            batched_y.append(torch.tensor([float(y)]))
        else:
            batched_y.append(y.unsqueeze(0) if y.dim() == 0 else y)
        
        # Shift edge indices by node_offset and collect
        if edge_index.shape[1] > 0:
            # edge_index is of shape (2, E)
            src = edge_index[0] + node_offset
            dst = edge_index[1] + node_offset
            batched_edge_index_src.append(src)
            batched_edge_index_dst.append(dst)
        
        # Update node offset for next graph
        node_offset += num_nodes
    
    # Concatenate node features
    batched_x = torch.cat(batched_x, dim=0) if batched_x else torch.tensor([])
    
    # Concatenate edge indices
    if batched_edge_index_src:
        batched_src = torch.cat(batched_edge_index_src, dim=0)
        batched_dst = torch.cat(batched_edge_index_dst, dim=0)
        batched_edge_index = torch.stack([batched_src, batched_dst])
    else:
        batched_edge_index = torch.tensor([[], []], dtype=torch.long)
    
    # Stack targets
    batched_y = torch.cat(batched_y, dim=0) if batched_y else torch.tensor([])
    
    # Convert batch to tensor
    batched_batch = torch.tensor(batched_batch, dtype=torch.long)
    
    return {
        'x': batched_x,
        'edge_index': batched_edge_index,
        'batch': batched_batch,
        'y': batched_y
    }

# Step 37 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    # TODO: Compute mean multi-class cross-entropy between logits and targets.
    # Use PyTorch's built-in cross entropy loss function
    # This computes log-softmax internally and returns the mean loss
    return F.cross_entropy(logits, targets, reduction='mean')

# Step 38 - mse_loss
import torch

def mse_loss(predictions, targets):
    # TODO: Compute mean squared error between predictions and targets
    # Flatten both tensors to 1-D
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Compute mean squared error
    return torch.mean((pred_flat - target_flat) ** 2)

# Step 39 - accuracy_metric
import torch

def accuracy_metric(logits, targets):
    # TODO: Return the fraction of argmax(logits) predictions matching targets.
    # Get predicted class indices
    predictions = torch.argmax(logits, dim=-1)
    
    # Count correct predictions and compute accuracy
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    
    return correct / total

# Step 40 - mae_metric
import torch

def mae_metric(predictions, targets):
    # TODO: Compute mean absolute error between predicted and target continuous values.
    # Flatten both tensors to 1-D
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Compute mean absolute error
    mae = torch.mean(torch.abs(pred_flat - target_flat))
    
    # Return as Python float
    return mae.item()

# Step 41 - gnn_train_step
import torch

def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    # TODO: Run one SGD training step and update params in-place...
    # Forward pass: compute predictions
    predictions = forward_fn(params, batch)
    
    # Compute loss
    loss = loss_fn(predictions, batch['y'])
    
    # Zero gradients (in case gradients exist from previous steps)
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()
    
    # Backward pass
    loss.backward()
    
    # Update parameters with SGD
    with torch.no_grad():
        for p in params.values():
            if p.grad is not None:
                p.sub_(lr * p.grad)
    
    # Return loss as Python float and updated params
    return {
        'loss': loss.item(),
        'params': params
    }

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

