/* Things TODO:
 * - Verbose mode
 * - Allow user to specify exported resolutions
 */

// retarget-lib
#include <retarget.hpp>

// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// C++ STL
#include <iostream>
#include <cstdlib>
#include <cstring>

struct MakeRetargetInput
{
    const char* saliency = nullptr;
    const char* color = nullptr;
    const char* output = nullptr;
};

void print_header();
void print_howto();
bool parse_args(int argc, char** argv, MakeRetargetInput& input);
bool make_retarget(const MakeRetargetInput& input);

int main(int argc, char* argv[])
{
    if (argc < 3 )
    {
        print_howto();
        exit(EXIT_FAILURE);
    }

    print_header();

    MakeRetargetInput input;
    if (!parse_args(argc, argv, input))
    {
        std::cerr << std::endl;
        print_howto();
        std::cerr << std::endl;
        std::cerr << "Invalid arguments, see above for more info" << std::endl;
        return EXIT_FAILURE;
    }

    if (!make_retarget(input))
    {
        std::cerr << "Failed to make-retarget, see above for more info" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void print_header()
{
    std::cout << "retarget-make" << std::endl;
}

void print_howto()
{
    std::cerr << "Usage: " << std::endl;
    std::cerr << "retarget-make -i<input> -s<saliency> [-o<output>]" << std::endl << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "-s<saliency> Saliency map associated with <input>" << std::endl;
    std::cerr << "-i<input> Input color map to be made retargetable" << std::endl;
    std::cerr << "-o<output> Output destination file, if not specified <input> will be overwritten" << std::endl << std::endl;
    std::cerr << "Notes: " << std::endl;
    std::cerr << "<saliency> and <input> are required to have the same resolution" << std::endl;
    std::cerr << "Ordering of argument is not important" << std::endl;
    std::cerr << "Most common image formats are supported see stb_image for saliency and adobe xmp for color " << std::endl;
    std::cerr << "Saliency currently evaluates only the first channel" << std::endl;
}

bool parse_args(int argc, char** argv, MakeRetargetInput& input)
{
    const char kInputSwitch[] = "-i";
    const char kSaliencySwitch[] = "-s";
    const char kOutputSwitch[] = "-o";

    input.color = nullptr;
    input.saliency = nullptr;
    input.output = nullptr;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strlen(argv[i]) < 2)
        {
            std::cerr << "Unrecognized argument: " << argv[i] << " skipping" << std::endl;
            continue;
        }
        char arg_switch[3];
        arg_switch[0] = argv[i][0]; arg_switch[1] = argv[i][1]; arg_switch[2] = '\0';
        char* arg = argv[i]+2;

        if (std::strcmp(arg_switch, kInputSwitch) == 0)
        {
            input.color = arg;
        }
        else if (std::strcmp(arg_switch, kSaliencySwitch) == 0)
        {
            input.saliency = arg;
        }
        else if (std::strcmp(arg_switch, kOutputSwitch) == 0)
        {
            input.output = arg;
        }
        else
            std::cerr << "Unrecognized arg: " << argv[i] << " skipping" << std::endl;
    }

    if (input.color == nullptr || input.saliency == nullptr)
    {
        std::cerr << "-i<input> and -s<saliency> are both required arguments" << std::endl;
        return false;
    }

/*    std::cout << "Input color: " << input.color << std::endl;
    std::cout << "Saliency: " << input.saliency << std::endl;
    if (input.output != nullptr)
        std::cout << "Output: " << input.output << std::endl;
    else
        std::cout << "Output: " << input.color << std::endl;*/
    return true;
}

bool make_retarget(const MakeRetargetInput& input)
{
    assert(input.color != nullptr);
    assert(input.saliency != nullptr);

    int w, h, comp;
    unsigned char* saliency_pixels = stbi_load(input.saliency, &w, &h, &comp, 0);
    if (saliency_pixels == nullptr)
    {
        std::cerr << "Failed to open: " << input.saliency << std::endl;
        return false;
    }

    unsigned char* saliency = (unsigned char*)malloc(sizeof(unsigned char) * w * h);
    for (int i = 0u; i < w * h; ++i)
    {
        saliency[i] = saliency_pixels[i * comp];
    }

    retarget::Vector2u resolutions []
    {
            { 1000, 200 },
            { 1000, 1000 },
            { 1000, 2000 },
            { 1000, 500 },
            { 1000, 250 },
            { 1000, 125 }
    };

    unsigned int num_resolutions = sizeof(resolutions) / sizeof(retarget::Vector2u);

    retarget::Metadata metadata;
    metadata.grid_size.x = 25;
    metadata.grid_size.y = 25;

    retarget::ResultCode res = retarget::ResultCode::Ok;
    const char* output = nullptr;

    metadata.entries = (retarget::MetadataEntry*)malloc(sizeof(retarget::MetadataEntry) * num_resolutions);
    metadata.num_entries = num_resolutions;
    double* spacings = (double*)malloc(sizeof(double) * num_resolutions * metadata.grid_size.x * metadata.grid_size.y);

    std::cout << "Generating: " << num_resolutions << " spacings" << std::endl;
    for (unsigned int i = 0u; i < num_resolutions; ++i)
    {
        double* cur_spacings = spacings + i * (metadata.grid_size.x + metadata.grid_size.y);
        res = retarget::calculate_spacings_25x25({ (unsigned int)w, (unsigned int)h },
                                                                      resolutions[i], saliency, cur_spacings);
        if (res != retarget::ResultCode::Ok)
        {
            std::cerr << "Failed to calculate spacings for resolution: " << w << "x" << h << " with error code: " << (int)res << std::endl;
            goto error;
        }

        metadata.entries[i].dimensions = resolutions[i];
        metadata.entries[i].spacings = spacings + i * (metadata.grid_size.x + metadata.grid_size.y);

        std::cout << "Generated: " << resolutions[i].x << "x" << resolutions[i].y << std::endl;
    }

    output = input.output;
    if (output == nullptr) output = input.color;

    res = retarget::encode_metadata(input.color, metadata, output);
    if (res != retarget::ResultCode::Ok)
    {
        std::cerr << "Failed to encode metadata with error: " << (int)res << std::endl;
        goto error;
    }

    free(metadata.entries);
    free(spacings);
    free(saliency);
    free(saliency_pixels);
    return true;
error:
    free(metadata.entries);
    free(spacings);
    free(saliency);
    free(saliency_pixels);
    return false;
}