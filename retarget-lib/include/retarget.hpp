#pragma once

// c++lib
#include <cstdlib>

namespace retarget
{
    /*
        Type: Spacing
            Represents the width of a grid's cell or the spacing between one line and the
            precedent / successive.
    */
    using Spacing = double;

    /*
        Type: Vector2s
            Simple 2D vector of uints used to represents 2D sizes.
    */
    struct Vector2u
    {
        unsigned int x;
        unsigned int y;
    };

    /*
        Type: MetadataEntry
            Single entry for a image's metadata contains grid data for the target
            <dimensions>. Note that the size of the grid is not specified in the entry
            because it's shared by all the entries associated with metadata block.

            dimensions - Resolution the grid has been generated for.
            spacings_x - Spacings for horizontal + vertical cells
        ( len depends on containing Metadata)
     */
    struct MetadataEntry
    {
        Vector2u dimensions;
        Spacing* spacings;
    };

    /*
        Type: Metadata
            Represents metadata to be encoded to an image file. All the entries have the same
            grid size for now.
     */
    struct Metadata
    {
        Vector2u grid_size;
        MetadataEntry* entries;
        size_t num_entries;
    };

    enum class ResultCode
    {
        Ok,
        InvalidArgument,
        InvalidArgumentSize,
        InternalEncoderError,
        FileNotFound,
        MetadataTooLarge
    };

    /*
        Function: calculate_spacings_25x25
            Calculates spacings for a 25x25 grid given the full resolution saliency map.
            This is usually loaded from a file. Currently only 8-bit single channel data
            is accepted.
            TODO: Add support for some basic image preprocessing including:
            > 16, 32 bits per channel
            > Multiple channels with pixel_data_step_rate, if for instance the saliency
                is encoded in the alpha channel.
            > Col-major

        src_size      - Size in pixels of the source image and also indicates the number
        of bytes for in_saliency
        dest_size     - Size in pixels of the destination image the grid has to be generated
        for.
        in_saliency   - Row-major saliency data.
        out_spacings  - Buffer of 25 + 25 where the data will be written to.
     */
    ResultCode calculate_spacings_25x25(Vector2u src_size,
                                        Vector2u dest_size,
                                        const unsigned char* in_saliency,
                                        Spacing* out_spacings);
    /*
        Function: encode_metadata
            Encodes metadata in the specified <src_image> and writes everything to
            <out_image>

            src_image  - Filename for the image to be read.
            metadata   - Metadata to be encoded.
            out_image  - If nullptr <src_image> is overwritten otherwise everything
                is written to a new file.
     */
    ResultCode encode_metadata(const char* src_image,
                               const Metadata& metadata,
                               const char* out_image = nullptr);
}
