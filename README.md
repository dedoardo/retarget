# !!WIP!! Still writing documentation and fixing bugs

# `retarget.js` 
Retarget.js is a script that enables image retargeting in browsers. Once included in a webpage it looks for custom metadata inside the images and adaptively retargets them to any resolution using WebGL.

The image is retargeted in real-time using an axis-aligned approach as described in [1](https://dl.acm.org/citation.cfm?id=2318863). Retargeting the image involves solving a constrained quadratic program. Although this could be implemented in Javascript, it would still add a significant overhead, linear in the number of images. For each image we solve for a variety of resolutions and encode the resulting grid as metadata in the images. This is achieved by `retarget.cpp`.  
When retargeting we then interpolate the closest aspect ratios to compute a grid transformation for any desired target resolution.

### Demo: You can find a demo at: [sparkon.github.io/retarget](https://sparkon.github.io/retarget/)

# `retarget.cpp`
`retarget.cpp` takes care of generating transformations for different resolutions and embedding them in the images. The metadata is embedded as XMP, we have noticed that the maximum XMP packet segment of 65kb is enough for an acceptable number of resolution. Extended xmp packets can also be encoded using the official Adobe XMP Toolkit [BSD-like] as Exiv2 apparently does not allow for encoding multiple XMP packets. 

The only input required to the process is a __saliency map__ which marks the important regions of the image. It is only used offline, it is not used by `retarget.js` in any way.

Saliency maps can be painted by hand, but if you want to generate them automatically, you can find some state-of-the-art research projects at Ming-Ming Cheng's excellent repository[2](https://github.com/MingMingCheng/CmCode). Other Open-source software can be found here [3](https://github.com/the-grid/gmr-saliency). 

# Builds
See the [Releases page](https://github.com/sparkon/retarget/releases) for the latest Windows x64 builds.  
If you are on Linux, OSX or elsewhere, compiling it yourself is quite easy. Read the instruction on the Compilation section.

# Usage
`./retarget --help`
```
./retarget <input_image> <input_saliency> <resolutions.txt>  <out_image>
    - <input_image> Source images
    - <input_saliency> Saliency map (should be the same resolutions as <input_image>)
    - [auto|<resolution>] file containing the resolutions to be exported. one per line with format %d %d\n
    - <output_image> Can be the same as the source one, it will get overwritten, but the properties will be preserved
```
- I recommend you use `auto` for the resolutions, it will generate a set of predefined target resolutions which should be enough for most usages.
- If you paint the saliency map by hand, I would recommend being conservative and utilizing the whole [0.0 - 1.0] range, not just 100% white.

# Compile instructions
`retarget.cpp` depends on:
- [Eigen](http://eigen.tuxfamily.org)
- [Exiv2](http://www.exiv2.org/)
- `stb_image.h` Included
They are both extremely well supported libraries: Eigen is header-only and Exiv2 has binaries for all platforms.  
For **Windows** A Visual Studio 2017 project is provided in the `retarget` directory, it assumes that both libraries are present in the same project directory.  
For **Linux** both Eigen and exiv2 are available on most package managers. 

# References:
[1] Daniele Panozzo, Ofir Weber, and Olga Sorkine. 2012. Robust Image Retargeting via Axis-Aligned Deformation. Comput. Graph. Forum 31, 2pt1 (May 2012), 229-236. DOI=http://dx.doi.org/10.1111/j.1467-8659.2012.03001.x  
[2] https://github.com/MingMingCheng/CmCode  
[3] https://github.com/the-grid/gmr-saliency  
