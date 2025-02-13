import tifffile
import numpy as np

# Read the TIFF file
tif_path = 'data.tif'
with tifffile.TiffFile(tif_path) as tif:
    # Get basic metadata
    print(f"Number of pages: {len(tif.pages)}")
    print("\nFirst page metadata:")
    page = tif.pages[0]
    print(f"Shape: {page.shape}")
    print(f"Dtype: {page.dtype}")
    print(f"Axes: {page.axes}")
    
    # Read the data
    data = tif.asarray()
    print(f"\nFull data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {data.min()}")
    print(f"Max value: {data.max()}")
    print(f"Mean value: {data.mean():.2f}")
    
    # Print first metadata tag if available
    print("\nFirst metadata tag:")
    tags = list(tif.pages[0].tags.values())
    if tags:
        first_tag = tags[0]
        print(f"{first_tag.name}: {first_tag.value}") 