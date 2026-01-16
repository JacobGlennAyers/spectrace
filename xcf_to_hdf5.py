#!/usr/bin/env python3
"""
One-time conversion of all XCF projects to ML-ready HDF5 format.
"""
import sys
from pathlib import Path
from ml_prep import XCFToStandardConverter

def main():
    # Your existing project folder
    project_folder = "projects"
    
    # New folder for ML data
    ml_data_folder = "hdf5_files"
    
    # Optional: path to master template
    template_xcf = "templates/orca_template.xcf"  # or None
    
    print("Starting conversion...")
    print(f"Input: {project_folder}")
    print(f"Output: {ml_data_folder}")
    
    # Create converter
    converter = XCFToStandardConverter(
        project_folder=project_folder,
        output_folder=ml_data_folder,
        layer_group_name="OrcinusOrca_FrequencyContours",
        format="hdf5",  # or "npz"
        color_mapping=None  # Will auto-load from project_folder
    )
    
    # Run conversion
    index_df = converter.convert_all()
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total samples: {len(index_df)}")
    print(f"   Index saved to: {ml_data_folder}/dataset_index.csv")
    
    # Print summary statistics
    if len(index_df) > 0:
        print("\n📊 Dataset Summary:")
        # Show available columns first
        print(f"   Available columns: {list(index_df.columns)}")
        
        # Show relevant columns if they exist
        cols_to_show = ['clip_basename', 'project_index', 'has_annotations']
        available_cols = [col for col in cols_to_show if col in index_df.columns]
        
        if available_cols:
            print(index_df[available_cols].head(10))
        else:
            print(index_df.head(10))
    else:
        print("\n⚠️  No samples were successfully converted!")
        print("   Check the error messages above for details.")
    
    return index_df

if __name__ == "__main__":
    index_df = main()
