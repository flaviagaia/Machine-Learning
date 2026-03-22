from __future__ import annotations

from src.knn_profiles import MODEL_PATH, NEIGHBORS_PATH, SUMMARY_PATH, build_profile_engine


def main() -> None:
    summary = build_profile_engine()

    print("Customer Segmentation with KNN Profiles")
    print("-" * 40)
    print(f"Customers: {summary['n_customers']}")
    print(f"Segments: {summary['n_segments']}")
    print(f"Silhouette score: {summary['silhouette_score']:.3f}")
    print(f"Sample customer: {summary['sample_customer_id']} ({summary['sample_customer_segment']})")
    print("Nearest neighbors:")
    for item in summary["sample_neighbors"]:
        print(f"  - {item['customer_id']} | {item['segment_label']} | distance={item['distance']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Neighbors saved to: {NEIGHBORS_PATH}")


if __name__ == "__main__":
    main()
