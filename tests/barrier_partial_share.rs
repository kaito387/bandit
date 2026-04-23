use bandit::{BarrierPartialSharePolicy, TreeConfig};

#[test]
fn safe_nodes_follow_shared_subtree_rule() {
    let policy = BarrierPartialSharePolicy::new(TreeConfig::new(2, 2, vec![0]), 0.01, 0.01, 1);

    let root = policy.tree.root;
    let shared_internal = policy.tree.subtree_root_from_path(&[0]).unwrap();
    let risky_internal = policy.tree.subtree_root_from_path(&[1]).unwrap();
    let shared_leaf = policy.tree.subtree_root_from_path(&[0, 0]).unwrap();

    assert!(!policy.tree.node(root).is_safe);
    assert!(policy.tree.node(shared_internal).is_safe);
    assert!(!policy.tree.node(risky_internal).is_safe);
    assert!(policy.tree.node(shared_leaf).is_safe);
}

#[test]
fn leaf_counts_match_subtree_leaf_numbers() {
    let policy = BarrierPartialSharePolicy::new(TreeConfig::new(2, 2, vec![0]), 0.01, 0.01, 1);

    let root = policy.tree.root;
    let shared_internal = policy.tree.subtree_root_from_path(&[0]).unwrap();
    let risky_internal = policy.tree.subtree_root_from_path(&[1]).unwrap();
    let shared_leaf = policy.tree.subtree_root_from_path(&[0, 0]).unwrap();

    assert_eq!(policy.tree.node(root).leaf_count, 4);
    assert_eq!(policy.tree.node(shared_internal).leaf_count, 2);
    assert_eq!(policy.tree.node(risky_internal).leaf_count, 2);
    assert_eq!(policy.tree.node(shared_leaf).leaf_count, 1);
    assert_eq!(policy.tree.node(shared_leaf).w, 1.0);
}

#[test]
fn sampling_and_updates_propagate_to_root() {
    let mut policy = BarrierPartialSharePolicy::new(TreeConfig::new(2, 2, vec![0]), 0.0, 1.0, 1);

    policy.tree.node_mut(0).theta[0] = 10.0;

    let before_root = policy.tree.node(0).w;
    let trace = policy.sample_round();
    let selected_child = trace.path_nodes[1];
    let before_child = policy.tree.node(selected_child).w;

    assert_eq!(trace.path_nodes.len(), 3);
    assert_eq!(trace.reach_probs[0], 1.0);

    policy.apply_observation(&trace, 0.0);

    assert!(policy.tree.node(0).w >= before_root);
    assert!(policy.tree.node(selected_child).w >= before_child);
}

#[test]
fn risky_theta_updates_use_path_probability() {
    let mut policy = BarrierPartialSharePolicy::new(TreeConfig::new(2, 2, vec![1]), 0.0, 1.0, 3);

    policy.tree.node_mut(0).theta[0] = 8.0;
    policy.tree.node_mut(0).theta[1] = 0.0;

    let trace = policy.sample_round();
    let leaf_before = policy.tree.node(trace.path_nodes[2]).w;
    let theta_before = policy.tree.node(trace.path_nodes[0]).theta.clone();

    policy.apply_observation(&trace, 1.0);

    let leaf_after = policy.tree.node(trace.path_nodes[2]).w;
    let theta_after = policy.tree.node(trace.path_nodes[0]).theta.clone();

    assert_eq!(leaf_after, leaf_before);
    assert!(theta_after != theta_before);
}
