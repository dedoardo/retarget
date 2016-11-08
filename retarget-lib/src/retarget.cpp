// Header
#include "../include/retarget.hpp"

// convex optimizer
#include "cvxgen_solver25x25/solver_25x25.h"

// c++lib
#include <assert.h>
#include <string>

// XMP Defines
#define TXMP_STRING_TYPE std::string
#define XMP_INCLUDE_XMPFILES 1

// XMP Includes
#include <XMP.incl_cpp>
#include <XMP.hpp>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#undef min
#undef max
#endif

namespace retarget
{
    double min(double a, double b) { return a < b ? a : b; }
    double max(double a, double b) { return a > b ? a : b; }

    // TODO: Move operators
    struct Matrix
    {
        Matrix(unsigned int rows, unsigned int cols, double val) :
                rows(rows), cols(cols), buffer(nullptr)
        {
            double* buf = new double[rows * cols];
            buffer = new double*[rows];
            for (unsigned int i = 0u; i < rows; ++i)
                buffer[i] = buf + i * cols;

            for (unsigned int i = 0u; i < rows * cols; ++i)
                buffer[0][i] = val;
        }

        ~Matrix()
        {
            if (buffer != nullptr)
            {
                delete[] buffer[0];
                delete[] buffer;
            }
        }

        Matrix(const Matrix& other) :
                rows(other.rows),
                cols(other.cols)
        {
            double* buf = new double[rows * cols];
            buffer = new double*[rows];
            for (unsigned int i = 0u; i < rows; ++i)
                buffer[i] = buf + i * cols;

            for (unsigned int i = 0u; i < rows * cols; ++i)
                buffer[0][i] = other.buffer[0][i];
        }

        Matrix& operator=(const Matrix& other) = delete;

        double& operator()(unsigned int row, unsigned int col)
        {
            assert(row < rows && col < cols);
            return buffer[row][col];
        }

        double operator()(unsigned int row, unsigned int col)const
        {
            assert(row < rows && col < cols);
            return buffer[row][col];
        }

        const unsigned int rows;
        const unsigned int cols;
        double** buffer;
    };

    Matrix transpose(const Matrix& matrix)
    {
        Matrix ret(matrix.cols, matrix.rows, 0.0);

        for (unsigned int i = 0u; i < matrix.rows; ++i)
        {
            for (unsigned int j = 0u; j < matrix.cols; ++j)
            {
                ret(j, i) = matrix(i, j);
            }
        }

        return ret;
    }

    Matrix mul(const Matrix& left, const Matrix& right)
    {
        Matrix ret(left.rows, right.cols, 0.);

        for (unsigned int i = 0u; i < left.rows; ++i)
        {
            for (unsigned int j = 0u; j < right.cols; ++j)
            {
                for (unsigned int k = 0u; k < left.cols; ++k)
                {
                    ret(i, j) += left(i, k) * right(k, j);
                }
            }
        }

        return ret;
    }

    // Thanks: http://www.adp-gmbh.ch/cpp/common/base64.html
    static const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789+/";

    inline std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
        std::string ret;
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];

        while (in_len--) {
            char_array_3[i++] = *(bytes_to_encode++);
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (i = 0; (i <4); i++)
                    ret += base64_chars[char_array_4[i]];
                i = 0;
            }
        }

        if (i)
        {
            for (j = i; j < 3; j++)
                char_array_3[j] = '\0';

            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (j = 0; (j < i + 1); j++)
                ret += base64_chars[char_array_4[j]];

            while ((i++ < 3))
                ret += '=';

        }

        return ret;
    }

    const char kBasePropertyName[]{ "xmp:RetargetSpacings" }; // + NxM
    const char kPropertyCellsX[]{ "xmp:RetargetCellsX" };
    const char kPropertyCellsY[]{ "xmp:RetargetCellsY" };

    ResultCode calculate_spacings_25x25(Vector2u src_size,
                                        Vector2u dest_size,
                                        const unsigned char* in_saliency,
                                        Spacing* out_spacings)
    {
        if (in_saliency == nullptr || out_spacings == nullptr)
            return ResultCode::InvalidArgument;

        if (src_size.x == 0 || src_size.y == 0 ||
            dest_size.x == 0 || dest_size.y == 0)
        {
            return ResultCode::InvalidArgumentSize;
        }

        constexpr unsigned int kCellsX = 25u;
        constexpr unsigned int kCellsY = 25u;
        constexpr double kDefaultMinW = 0.15;
        constexpr double kDefaultMinH = 0.15;
        const unsigned int pixels_per_cell_x = src_size.x/ kCellsX;
        const unsigned int pixels_per_cell_y = src_size.y / kCellsY;

        Matrix saliency(kCellsX, kCellsY, 0.0);

        // Integrating saliency
        // TODO: Can be improved by looping over src image instead of saliency?
        const unsigned int hf = src_size.y / kCellsY;
        const unsigned int wf = src_size.x / kCellsX;
        for (unsigned int y = 0u; y < kCellsY; ++y)
        {
            const unsigned int start_y = 0 + y * hf;
            const unsigned int end_y = 0 + (y + 1) * hf;

            for (unsigned int x = 0u; x < kCellsX; ++x)
            {
                double acc = 0.0;
                unsigned int counter = 0;

                const unsigned int start_x = 0 + x * wf;
                const unsigned int end_x = 0 + (x + 1) * wf;

                for (unsigned int py = start_y; py < end_y; ++py)
                {
                    for (unsigned int px = start_x; px < end_x; ++px)
                    {
                        acc += ::retarget::max(1.0, (double)in_saliency[py * src_size.x + px]);
                        ++counter;
                    }
                }

                saliency(y, x) = sqrt(acc / counter);
            }
        }

        // TODO: Make them tweakable
        double min_w = ::retarget::min(kDefaultMinW, (double)dest_size.x / kCellsX);
        double min_h = ::retarget::min(kDefaultMinH, (double)dest_size.y / kCellsY);

        // Generating ASAP energy system
        double s_factor = 1.0 / ((double)src_size.x / kCellsX);
        double t_factor = 1.0 / ((double)src_size.y / kCellsY);

        // K dim(M * N, M + N)
        Matrix K(kCellsX * kCellsY, kCellsX + kCellsY, 0.0);

        // Sparse matrix
        for (unsigned int i = 0u; i < kCellsX; ++i)
        {
            for (unsigned int j = 0u; j < kCellsY; ++j)
            {
                unsigned int k = i * kCellsX + j;

                K(k, i)			  = s_factor * saliency(j, i);
                K(k, j + kCellsX) = t_factor * -saliency(j, i);
            }
        }

        Matrix M = mul(transpose(K), K);

        // Q = K^t * K <= Q is positive semi-defined
        const unsigned int Q_size = (kCellsX + kCellsY) * (kCellsX + kCellsY);
        double* Q = new double[Q_size];
        for (unsigned int i = 0u; i < Q_size; ++i)
            Q[i] = 0.0;

        for (unsigned int j = 0u; j < M.cols; ++j)
            for (unsigned int i = 0u; i < M.rows; ++i)
            {
                Q[j * M.cols + i] = M(i, j);
            }

        // Preparing solver
        CVXGenImageResizing25x25 cvx_solver;
        cvx_solver.set_defaults();
        cvx_solver.settings.verbose = 0;
        cvx_solver.setup_indexing();

        cvx_solver.params.imageWidth[0] = (double)dest_size.x;
        cvx_solver.params.imageHeight[0] = (double)dest_size.y;
        cvx_solver.params.minLengthW[0] = min_w;
        cvx_solver.params.minLengthH[0] = min_h;

        for (unsigned int i = 0u; i < kCellsX + kCellsY; ++i)
        {
            for (unsigned int j = 0u; j < kCellsX + kCellsY; ++j)
            {
                cvx_solver.params.E[i * (kCellsX + kCellsY) + j] = Q[i * (kCellsX + kCellsY) + j];
            }
        }

        for (unsigned int i = 0u; i < kCellsX + kCellsY; ++i)
            cvx_solver.params.B[i] = 0.0;


        // Results will be in vars.st
        for (unsigned int i = 0; i < kCellsX + kCellsY; ++i)
            cvx_solver.vars.st[i] = 0.0;

        /*
        Inputs for the QP Solver are:
        Energy > B dim( M + N )
        Energy > Q dim( M + N, M + N )
        Lw/Lh are the min values for the grid spacings
        all rows >= Lw
        all cols >= Lh
        sum of M rows = dest_w
        sum of N cols = dest_h
        */
        cvx_solver.solve();

        delete[] Q;

        // Storing results
        for (unsigned int i = 0u; i < kCellsX + kCellsY; ++i)
            out_spacings[i] = cvx_solver.vars.st[i];

        return ResultCode::Ok;
    }

    ResultCode encode_metadata(const char* src_image,
                               const Metadata& metadata,
                               const char* out_image)
    {
        if (out_image != nullptr && strcmp(src_image, out_image) != 0)
        {
#if defined(_WIN32)
            CopyFileA(src_image, out_image, false);
#endif
        }

        if (out_image == nullptr)
            out_image = src_image;

        if (!SXMPMeta::Initialize())
            return ResultCode::InternalEncoderError;

        if (!SXMPFiles::Initialize(0x0))
        {
            SXMPMeta::Terminate();
            return ResultCode::InternalEncoderError;
        }

        try
        {
            XMP_OptionBits file_opts = kXMPFiles_OpenForUpdate | kXMPFiles_OpenUseSmartHandler;

            SXMPFiles file;
            auto open_res = file.OpenFile(out_image, kXMP_UnknownFile, file_opts);
            if (!open_res)
                return ResultCode::FileNotFound;

            SXMPMeta meta;
            file.GetXMP(&meta);

            // Writing number of cells
            if (meta.DoesPropertyExist(kXMP_NS_XMP, kPropertyCellsX))
                meta.DeleteProperty(kXMP_NS_XMP, kPropertyCellsX);

            if (meta.DoesPropertyExist(kXMP_NS_XMP, kPropertyCellsY))
                meta.DeleteProperty(kXMP_NS_XMP, kPropertyCellsY);

            meta.SetProperty(kXMP_NS_XMP, kPropertyCellsX, base64_encode(reinterpret_cast<const unsigned char*>(&metadata.grid_size.x), sizeof(uint16_t)));
            meta.SetProperty(kXMP_NS_XMP, kPropertyCellsY, base64_encode(reinterpret_cast<const unsigned char*>(&metadata.grid_size.y), sizeof(uint16_t)));

            for (unsigned int i = 0; i < metadata.num_entries; ++i)
            {
                // Encoding property in name ( TODO: can be changed )
                const auto& entry = metadata.entries[i];
                std::string property_name = kBasePropertyName + std::to_string(entry.dimensions.x) + 'x' + std::to_string(entry.dimensions.y);

                // Checking if property name exists, if not we create it, and if
                // it does exist already we destroy it and recreate ( to ensure types )
                if (meta.DoesPropertyExist(kXMP_NS_XMP, property_name.c_str()))
                    meta.DeleteProperty(kXMP_NS_XMP, property_name.c_str());

                // Recreating the property
                XMP_OptionBits create_opts = kXMP_PropValueIsArray | kXMP_PropArrayIsOrdered;
                for (unsigned int j = 0; j < metadata.grid_size.x + metadata.grid_size.y; ++j)
                {
                    // encoding and appending
                    auto encoded_value = base64_encode(reinterpret_cast<const unsigned char*>(&metadata.entries[i].spacings[j]), sizeof(double));
                    meta.AppendArrayItem(kXMP_NS_XMP, property_name.c_str(), j ? 0x0 : create_opts, encoded_value, 0x0);
                }
            }

            if (!file.CanPutXMP(meta))
            {
                SXMPMeta::Terminate();
                SXMPFiles::Terminate();
                return ResultCode::MetadataTooLarge;
            }

            file.PutXMP(meta);
            file.CloseFile();
        }
        catch (XMP_Error&) // TODO: Dump error
        {
            return ResultCode::InternalEncoderError;
        }

        SXMPMeta::Terminate();
        SXMPFiles::Terminate();

        return ResultCode::Ok;
    }
}