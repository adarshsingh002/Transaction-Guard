"""Layer 2 graph clustering using Louvain community detection."""
import os
import sys
import pandas as pd
import networkx as nx
import community as community_louvain

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.layer2.graph import update_cluster_cache

def refine_clusters(G: nx.Graph, initial_mapping: dict, initial_summary: pd.DataFrame):
    """
    Takes the initial connected components and refines them using Louvain community detection
    for components larger than LOUVAIN_MIN_COMPONENT_SIZE.
    """
    refined_mapping = {}
    refined_summary_list = []
    
    cluster_counter = 0
    
    for _, row in initial_summary.iterrows():
        members = row['members']
        size = row['size']
        
        if size >= config.LOUVAIN_MIN_COMPONENT_SIZE:
            # Refine using Louvain
            subgraph = G.subgraph(members)
            
            # community_louvain returns a dict mapping node -> community_id
            partition = community_louvain.best_partition(subgraph, weight='weight')
            
            # Group nodes by community
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
                
            for comm_id, comm_members in communities.items():
                new_cluster_id = f"refined_cluster_{cluster_counter}"
                cluster_counter += 1
                
                for acc in comm_members:
                    refined_mapping[acc] = new_cluster_id
                    
                comm_subgraph = G.subgraph(comm_members)
                shared_types = set()
                for u, v, data in comm_subgraph.edges(data=True):
                    shared_types.update(data.get('shared_types', []))
                    
                refined_summary_list.append({
                    'cluster_id': new_cluster_id,
                    'size': len(comm_members),
                    'shared_types': list(shared_types),
                    'members': comm_members
                })
        else:
            # Keep as is
            new_cluster_id = f"refined_cluster_{cluster_counter}"
            cluster_counter += 1
            
            for acc in members:
                refined_mapping[acc] = new_cluster_id
                
            refined_summary_list.append({
                'cluster_id': new_cluster_id,
                'size': size,
                'shared_types': row['shared_types'],
                'members': members
            })
            
    update_cluster_cache(refined_mapping)
    
    df_refined_summary = pd.DataFrame(refined_summary_list)
    return refined_mapping, df_refined_summary
