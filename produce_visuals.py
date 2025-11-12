from utils import visualize_all_projects, visualize_overlay, visualize_individual_layers

if __name__ == "__main__":
    # Option 1: Visualize all projects for a specific clip
    print("="*60)
    print("Generating visualizations for all projects...")
    print("="*60)
    
    visualize_all_projects(
        project_folder="./projects",
        clip_basename="2023-12-03--10-15-10--Rec-C.sel.26.AA",
        layer_group_name="OrcinusOrca_FrequencyContours",
        output_folder="./visualizations"
    )
    
    # Option 2: Visualize a single specific project
    # Uncomment the lines below to visualize just one project
    """
    print("\n" + "="*60)
    print("Generating visualizations for a single project...")
    print("="*60)
    
    project_path = "./projects/2023-12-03--10-15-10--Rec-C.sel.26.AA_0"
    
    # Create overlay visualization
    visualize_overlay(
        project_path=project_path,
        layer_group_name="OrcinusOrca_FrequencyContours",
        project_index=0,
        output_path="./single_overlay_viz.png"
    )
    
    # Create individual layers visualization
    visualize_individual_layers(
        project_path=project_path,
        layer_group_name="OrcinusOrca_FrequencyContours",
        output_path="./single_individual_viz.png"
    )
    """
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("="*60)