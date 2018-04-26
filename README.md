#### !WIP! Working on documentation / builds / examples

# `retarget.js` 
Retarget.js is a script that enables image retargeting in browsers. Once included in a webpage it looks for custom metadata inside the images and adaptively retargets them to any resolution using WebGL.

The image is retargeted in real-time using an axis-aligned approach as described in [1]. Retargeting the image involves solving a constrained quadratic program. Although this could be implemented in Javascript, it would still add a significant overhead, linear in the number of images. For each image we solve for a variety of resolutions and encode the resulting grid as metadata in the images. This is achieved by `retarget.cpp`.  
When retargeting we then interpolate the closest aspect ratios to compute a grid transformation for any desired target resolution.

### Demo: You can find a demo at: [sparkon.github.io/retarget](https://sparkon.github.io/retarget/)

# `retarget.cpp`
`retarget.cpp` takes care of generating transformations for different resolutions and embedding them in the images. The metadata is embedded as XMP, we have noticed that the maximum XMP packet segment of 65kb is enough for an acceptable number of resolution. Extended xmp packets can also be encoded using the official Adobe XMP Toolkit [BSD-like] as Exiv2 apparently does not allow for encoding multiple XMP packets. 

The only input required to the process is a __saliency map__ which indentifies the important region of the image. The saliency map is only used offline, it is not used by `retarget.js` in any way.

Saliency maps can be painted by hand, but if you want to generate them automatically, you can find some non-commercial state-of-the-art research projects at Ming-Ming Cheng's excellent repository[2]. Other Open-source software can be found here [3]. 
`retarget-auto.cpp` is an extension to the metadata embedder (`retarget.cpp`) which uses Ming-Ming Cheng's Contrast-Based approach[2] to automatically generate a saliency map. You can find a precompiled binary here <link>. If you want to compile it yourself there is a little extra work to be done, but a CMakeLists.txt is provided.  
<compilation instructions>

##### See the Download section for Windows builds.
##### Compiling on Linux/OSX is also pretty straighforward   


# Usage
TODO

# Compiling `retarget.cpp` 
TODO

# References:
[1] Daniele Panozzo, Ofir Weber, and Olga Sorkine. 2012. Robust Image Retargeting via Axis-Aligned Deformation. Comput. Graph. Forum 31, 2pt1 (May 2012), 229-236. DOI=http://dx.doi.org/10.1111/j.1467-8659.2012.03001.x
[2] https://github.com/MingMingCheng/CmCode
[3] https://github.com/the-grid/gmr-saliency
