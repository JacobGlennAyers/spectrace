from utils import (
    visualize_all_projects, 
    visualize_overlay, 
    visualize_individual_layers,
    get_or_create_color_mapping
)

if __name__ == "__main__":
    # Configuration
    project_folder = "./projects"
    clip_basename = "orca"
    layer_group_name = "OrcinusOrca_FrequencyContours"
    output_folder = "./visualizations"
    
    # Path to your master template XCF file (optional but recommended)
    # This file should contain all possible layer definitions for your species
    # Leave as None to auto-discover layers from all project XCF files instead
    template_xcf_path = "./templates/orca_template.xcf"  # or None
    
    # Step 1: Automatically discover layers and create/load color mapping
    # If template_xcf_path is provided, it reads layers from that master template
    # If None, it scans all project XCF files to discover layers
    print("="*60)
    print("🎨 Setting up color mapping...")
    print("="*60)
    color_mapping = get_or_create_color_mapping(
        project_folder, 
        layer_group_name,
        template_xcf_path=template_xcf_path
    )
    
    # Option 1: Visualize all projects for a specific clip
    print("\n" + "="*60)
    print("Generating visualizations for all projects...")
    print("="*60)
    visualize_all_projects(
        project_folder=project_folder,
        clip_basename=clip_basename,
        layer_group_name=layer_group_name,
        output_folder=output_folder,
        color_mapping=color_mapping,  # Pass the auto-generated color mapping
        template_xcf_path=template_xcf_path  # Pass template for consistency
    )
    
    # Option 2: Visualize a single specific project
    # Uncomment the lines below to visualize just one project
    """
    print("\n" + "="*60)
    print("Generating visualizations for a single project...")
    print("="*60)
    
    project_path = "./projects/test_0"
    
    # Create overlay visualization
    visualize_overlay(
        project_path=project_path,
        layer_group_name=layer_group_name,
        project_index=0,
        output_path="./single_overlay_viz.png",
        color_mapping=color_mapping  # Use the same color mapping
    )
    
    # Create individual layers visualization
    visualize_individual_layers(
        project_path=project_path,
        layer_group_name=layer_group_name,
        output_path="./single_individual_viz.png",
        color_mapping=color_mapping  # Use the same color mapping
    )
    """
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("="*60)