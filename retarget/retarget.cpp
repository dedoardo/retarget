// stdlib
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <algorithm>

// Eigen
#include "Eigen/Core"
#include "Eigen/LU"

// stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Eigen
#include "Eigen/Core"

// exiv2
#include <exiv2/exiv2.hpp>

using namespace std;
using namespace Eigen;

#define VERBOSE(x) x
#ifdef max
    #undef max
#endif

#ifdef min
    #undef min
#endif

struct RetargetArgs {
    enum class Energy {
        Similar,
        Rigid
    };

    Eigen::Vector2i grid_resolution;
    Eigen::MatrixXd saliency; // discretized saliency
    Eigen::Vector2d limits; // Minimum cell size
    Eigen::Vector2i src_resolution; // Original resolution
    Eigen::Vector2i dst_resolution; // Target resolution
    double          laplacian_weight;
    Energy          energy;
};

struct RetargetMetadata {
    struct Entry {
        Eigen::Vector2i resolution;
        Eigen::VectorXd cells;
    };

    std::vector<Entry> entries;
    Eigen::Vector2i resolution;
};

static bool read_resolutions ( const char* resolution_file, vector<Vector2i>& resolutions, Vector2i& grid_dimensions );
static void generate_auto_resolutions ( vector<Vector2i>& resolutions );
static bool retarget_image ( const RetargetArgs& args, RetargetMetadata::Entry& entry );
static bool integrate_saliency ( const char* filename, const Eigen::Vector2i& grid_resolution, Eigen::MatrixXd& integrated_saliency );
static bool encode_metadata ( const char* file, const RetargetMetadata& metadata );
static bool retarget_solve ( const RetargetArgs& args, Eigen::VectorXd& cell_dimensions, int* iters );

// Constants
const Vector2d DEFAULT_LIMITS             = Vector2d ( 0.75, 0.75 );
const double   DEFAULT_LAPLACIAN_WEIGHT   = 0.02;
const Vector2i DEFAULT_AUTORES_MIN        = Vector2i ( 400, 400 );
const Vector2i DEFAULT_AUTORES_MAX        = Vector2i ( 2000, 2000 );
const Vector2i DEFAULT_AUTORES_STEP       = Vector2i ( 400, 400 );
const Vector2i DEFAULT_GRID_SIZE          = Vector2i ( 35, 35 );
const RetargetArgs::Energy DEFAULT_ENERGY = RetargetArgs::Energy::Similar;

/*
    Command line API
    <input_image> <input_saliency> <resolutions.txt>  <out_image>
        - <irput_image> Source images
        - <input_saliency> Saliency map (should e the same resolutions)
        - <resolution> file containing the resolutions to be exported. one per line with format %d %d\n
        - <output_image> Can be the same as the source one, it will get overwritten, but the properties will be preserved
  */
int main ( int argc, const char* argv[] ) {
    if ( argc < 5 ) {
        cerr << "<input_image> <input_saliency> <resolutions.txt> <output_image>";
        return EXIT_FAILURE;
    }

    const char* input_image = argv[1];
    const char* input_saliency = argv[2];
    const char* input_resolutions = argv[3];
    const char* output_image = argv[4];
    vector<Vector2i> resolutions;

    Vector2i src_resolution;

    if ( !stbi_info ( input_image, &src_resolution.x(), &src_resolution.y(), nullptr ) ) {
        cerr << "Failed to read input image" << input_image <<  endl;
        return EXIT_FAILURE;
    }

    RetargetMetadata metadata;
    RetargetArgs args;
    args.src_resolution = src_resolution;
    args.laplacian_weight = DEFAULT_LAPLACIAN_WEIGHT;
    args.limits = DEFAULT_LIMITS;

    if ( strcmp ( input_resolutions, "auto" ) == 0 ) {
        generate_auto_resolutions ( resolutions );
        args.grid_resolution = DEFAULT_GRID_SIZE;
        args.energy = DEFAULT_ENERGY;
    } else if ( !read_resolutions ( input_resolutions, resolutions, args.grid_resolution ) ) {
        cerr << "Failed to read resolutions file" << endl;
        return EXIT_FAILURE;
    }

    if ( !integrate_saliency ( input_saliency, args.grid_resolution, args.saliency ) ) {
        cerr << "Failed to integrate saliency " << input_saliency << endl;
        return EXIT_FAILURE;
    }

    std::vector<double> aspect_ratios;

    for ( auto resolution : resolutions ) {
        double aspect_ratio = ( double ) resolution.x() / resolution.y();

        bool already_exported = false;

        std::for_each ( aspect_ratios.begin(), aspect_ratios.end(), [&] ( const double & ar ) {
            if ( abs ( ar - aspect_ratio ) < 1e-3 ) {
                already_exported = true;
            }
        } );

        if ( already_exported ) {
            continue;
        }

        args.dst_resolution = resolution;

        RetargetMetadata::Entry entry;

        if ( !retarget_image ( args, entry ) ) {
            cerr << "Failed to retarget image W=" << resolution.x() << " H=" << resolution.y() << endl;
            return EXIT_FAILURE;
        }

        VERBOSE ( cerr << "Calculated " << resolution.x() << " " << resolution.y() << endl );
        metadata.entries.push_back ( entry );
        aspect_ratios.push_back ( aspect_ratio );
    }

    metadata.resolution = args.grid_resolution;

    // Duplicating image before adding metadata if source != target
    if ( strcmp ( input_image, output_image ) != 0 ) {
        ifstream  src ( input_image, std::ios::binary );
        ofstream  dst ( output_image, std::ios::binary );
        dst << src.rdbuf();
    }

    if ( !encode_metadata ( output_image, metadata ) ) {
        cerr << "Failed to encode retargeting metadata in " << output_image << endl;
        return EXIT_FAILURE;
    }

    std::cin.get();

    return EXIT_SUCCESS;
}

bool read_resolutions ( const char* filename, std::vector<Vector2i>& resolutions, Vector2i& grid_dimensions ) {
    ifstream fs ( filename );
    resolutions.clear();
    string line;
    int i = 0;

    while ( getline ( fs, line ) ) {
        int w, h;

        if ( sscanf ( line.c_str(), "%d %d", &w, &h ) != 2 ) {
            cerr << "Skipping line " << i << ", invalid format." << endl;
            continue;
        }

        // First line is the grid resolution
        if ( i == 0 ) {
            grid_dimensions = Vector2i ( w, h );

        } else {
            resolutions.push_back ( Vector2i ( w, h ) );
            VERBOSE ( cout << "Adding resolution " << w << " " << h << endl );
        }

        ++i;
    }

    return true;
}

void generate_auto_resolutions ( vector<Vector2i>& resolutions ) {
    for ( int x = DEFAULT_AUTORES_MIN.x(); x <= DEFAULT_AUTORES_MAX.x(); x += DEFAULT_AUTORES_STEP.x() ) {
        for ( int y = DEFAULT_AUTORES_MIN.y(); y <= DEFAULT_AUTORES_MAX.y(); y += DEFAULT_AUTORES_STEP.y() ) {
            resolutions.push_back ( Vector2i ( x, y ) );
        }
    }
}

bool retarget_image ( const RetargetArgs& args, RetargetMetadata::Entry& entry ) {
    if ( !retarget_solve ( args, entry.cells, nullptr ) ) {
        return false;
    }

    entry.resolution = args.dst_resolution;
    assert ( entry.cells.size() == args.grid_resolution.x() + args.grid_resolution.y() );
    return true;
}

bool integrate_saliency ( const char* filename, const Eigen::Vector2i& grid_resolution, Eigen::MatrixXd& integrated_saliency ) {
    int w, h, n;
    stbi_uc* saliency = stbi_load ( filename, &w, &h, &n, 3 );

    if ( saliency == nullptr ) {
        fprintf ( stderr, "Failed lo load file: %s", filename );
        return false;
    }

    int cx = grid_resolution.x();
    int cy = grid_resolution.y();
    integrated_saliency = MatrixXd ( cx, cy );
    const int wf = w / cx;
    const int hf = h / cy;

    for ( int y = 0; y < cy; ++y ) {
        const int start_y = y * hf;
        const int end_y = ( y + 1 ) * hf;

        for ( int x = 0; x < cx; ++x ) {
            double acc = .0;
            int counter = 0;
            const int start_x = x * wf;
            const int end_x = ( x + 1 ) * wf;

            for ( int py = start_y; py < end_y; ++py ) {
                for ( int px = start_x; px < end_x; ++px ) {
                    uint8_t* d = saliency + ( py * w + px ) * 3;
                    acc += ::std::max ( 0.0001, ( ( double ) d[0] + ( double ) d[1] + ( double ) d[2] ) / ( 255 * 3 ) );
                    ++counter;
                }
            }

            integrated_saliency ( y, x ) = ::std::sqrt ( acc / counter );
        }
    }

    stbi_image_free ( saliency );
    return true;
}

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline std::string base64_encode ( unsigned char const* bytes_to_encode, unsigned int in_len ) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while ( in_len-- ) {
        char_array_3[i++] = * ( bytes_to_encode++ );

        if ( i == 3 ) {
            char_array_4[0] = ( char_array_3[0] & 0xfc ) >> 2;
            char_array_4[1] = ( ( char_array_3[0] & 0x03 ) << 4 ) + ( ( char_array_3[1] & 0xf0 ) >> 4 );
            char_array_4[2] = ( ( char_array_3[1] & 0x0f ) << 2 ) + ( ( char_array_3[2] & 0xc0 ) >> 6 );
            char_array_4[3] = char_array_3[2] & 0x3f;

            for ( i = 0; ( i < 4 ); i++ ) {
                ret += base64_chars[char_array_4[i]];
            }

            i = 0;
        }
    }

    if ( i ) {
        for ( j = i; j < 3; j++ ) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = ( char_array_3[0] & 0xfc ) >> 2;
        char_array_4[1] = ( ( char_array_3[0] & 0x03 ) << 4 ) + ( ( char_array_3[1] & 0xf0 ) >> 4 );
        char_array_4[2] = ( ( char_array_3[1] & 0x0f ) << 2 ) + ( ( char_array_3[2] & 0xc0 ) >> 6 );
        char_array_4[3] = char_array_3[2] & 0x3f;

        for ( j = 0; ( j < i + 1 ); j++ ) {
            ret += base64_chars[char_array_4[j]];
        }

        while ( ( i++ < 3 ) ) {
            ret += '=';
        }
    }

    return ret;
}

static bool encode_metadata ( const char* file, const RetargetMetadata& metadata ) {
    const string TAG_CELLS_X = "Xmp.Retarget.CellsX";
    const string TAG_CELLS_Y = "Xmp.Retarget.CellsY";
    const string TAG_SPACINGS = "Xmp.Retarget.Spacings";

    try {
        vector<string> xmp_packets;
        Exiv2::XmpProperties::registerNs ( "Retarget/", "Retarget" );

        // Creating metadata packet
        Exiv2::XmpData xmp;

        uint16_t cells_x = metadata.resolution.x();
        auto cells_x_v = Exiv2::Value::create ( Exiv2::xmpText );
        cells_x_v->read ( base64_encode ( ( unsigned char* ) &cells_x, 2 ) );
        xmp.add ( Exiv2::XmpKey ( TAG_CELLS_X ), cells_x_v.get() );

        uint16_t cells_y = metadata.resolution.y();
        auto cells_y_v = Exiv2::Value::create ( Exiv2::xmpText );
        cells_y_v->read ( base64_encode ( ( unsigned char* ) &cells_y, 2 ) );
        xmp.add ( Exiv2::XmpKey ( TAG_CELLS_Y ), cells_y_v.get() );

        int last_serialized = -1;
        int to_serialize = 0;
        string last_packet = "";

        for ( const RetargetMetadata::Entry& entry : metadata.entries ) {

            std::string property_name = TAG_SPACINGS + std::to_string ( entry.resolution.x() ) + 'x' + std::to_string ( entry.resolution.y() );

            auto xmp_seq = Exiv2::Value::create ( Exiv2::xmpSeq );

            for ( int i = 0; i < entry.cells.size(); ++i ) {
                xmp_seq->read ( base64_encode ( ( unsigned char* ) &entry.cells[i], sizeof ( double ) ) );
            }

            xmp.add ( Exiv2::XmpKey ( property_name ), xmp_seq.get() );

            // We try to serialize the packet each time and see how much space left we have
            string cur_packet;

            if ( Exiv2::XmpParser::encode ( cur_packet, xmp ) != 0 ) {
                cerr << "Internal xmp error. Contact the mantainer." << endl;
                return false;
            }

            VERBOSE ( cerr << "Current packet size: " << cur_packet.size() << endl );

            // 65kb is the JPEG xmp packet limit size
            if ( cur_packet.size() >= std::numeric_limits<uint16_t>::max() ) {
                // Do we have at least one set of spacings ?
                if ( last_serialized == to_serialize ) {
                    cerr << "Minimum retarget metadata unit doesn't fit in xmp packet size. Contact the mantainer." << endl;
                    return false;
                }

                assert ( !last_packet.empty() );
                xmp_packets.push_back ( last_packet );
                last_packet = "";
                xmp.clear();
            } else {
                last_packet = cur_packet;
            }

            ++to_serialize;
        }

        if ( !last_packet.empty() ) {
            xmp_packets.push_back ( last_packet );
        }

        // Opening image
        Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open ( file );

        if ( image.get() == 0 ) {
            fprintf ( stderr, "Failed to open image %s for writing metadata", file );
            return false;
        }

        // Writing metadata
        for ( const auto& packet : xmp_packets ) {
            image->setXmpPacket ( packet );
        }

        image->writeMetadata();
    } catch ( Exiv2::AnyError& e ) {
        cerr << "Exiv2 Error: " << e << "'\n";
        return false;
    }

    return true;
}

static void export_matrix ( const string& filename, const MatrixXd& matrix ) {
#ifdef DEBUG_MATRICES
    std::ofstream fs ( std::string ( "matrix/" ) + filename + ".txt" );
    fs << "Dimension: " << matrix.rows() << "x" << matrix.cols() << std::endl;
    fs << "Max Coeff: " << matrix.maxCoeff() << " Min Coeff: " << matrix.minCoeff() << std::endl;
    fs << matrix;
    std::vector<uint8_t> pixels ( matrix.rows() * matrix.cols(), 0 );
    MatrixXd matrix_t = matrix.transpose();

    for ( int i = 0; i < matrix.size(); ++i ) {
        if ( abs ( matrix_t.data() [i] ) < 1e-15 ) {
            pixels[i] = 0;
        } else {
            pixels[i] = 255;
        }
    }

    if ( !stbi_write_png ( ( std::string ( "matrix/" ) + filename + ".png" ).c_str(), matrix.cols(), matrix.rows(), 1, pixels.data(), matrix.cols() ) ) {
        fprintf ( stderr, "Failed to dump matrix to file: %s", filename );
    }

#endif
}

static void compute_matrices ( const RetargetArgs& args, MatrixXd& K, VectorXd& k ) {
    // We are trying to solve
    //  x'Qx + x'b
    // for As-Similar-As-Possible deformations we have
    // b = 0
    // Q = K'K
    // where K is calculated as in equation (7)
    const int nrows = args.grid_resolution.x();
    const int ncols = args.grid_resolution.y();
    const double min_grid_dim = 0.15;
    const double min_w = ::std::min ( min_grid_dim, ( double ) args.dst_resolution.x() / ncols );
    const double min_h = ::std::min ( min_grid_dim, ( double ) args.dst_resolution.y() / nrows );
    const double s_factor = nrows / ( double ) args.src_resolution.y();
    const double t_factor = ncols / ( double ) args.src_resolution.x();

    switch ( args.energy ) {
        case RetargetArgs::Energy::Similar: {
            // ASAP
            K = MatrixXd::Constant ( ncols * nrows, ncols + nrows, 0.0 );
            export_matrix ( "ASAP_K", K );

            for ( int i = 0; i < ncols; ++i ) {
                for ( int j = 0; j < nrows; ++j ) {
                    int k = i * ncols + j;
                    double s = ::std::max ( args.saliency ( j, i ), 0.05 );
                    //double s = 1.0;
                    K ( k, i ) = s_factor * s;
                    K ( k, j + ncols ) = t_factor * -s;
                }
            }

            k = VectorXd::Constant ( ncols * nrows, 0 );
            export_matrix ( "ASAP_k", k );
            break;
        }

        case RetargetArgs::Energy::Rigid: {

            break;
        }
    }

    // Laplacian Regularization
    int newcountx = ncols - 1;
    int newcounty = nrows - 1;
    int newcount = newcountx + newcounty;
    int startx = ( int ) K.rows();
    int starty = ( int ) K.rows() + newcountx;

    MatrixXd Kext = MatrixXd::Zero ( K.rows() + newcount, K.cols() );
    Kext.block ( 0, 0, K.rows(), K.cols() ) = K;
    K = Kext;

    VectorXd kext = VectorXd::Zero ( k.size() + newcount );
    kext.head ( k.size() ) = k;
    k = kext;

    for ( int i = 0; i < newcountx; ++i ) {
        K ( i + startx, i ) = args.laplacian_weight;
        K ( i + startx, i + 1 ) = -args.laplacian_weight;
    }

    for ( int i = 0; i < newcounty; ++i ) {
        K ( i + starty, i + ncols ) = args.laplacian_weight;
        K ( i + starty, i + ncols + 1 ) = -args.laplacian_weight;
    }
}


#if 1
// Primal-dual path following. Using Mehortra's Predictor-Corrector approach
// Using a similar derivation as in http://stanford.edu/~boyd/papers/code_gen_impl.html
static bool retarget_solve ( const RetargetArgs& args, VectorXd& solution, int* iters ) {
    // Inputs
    MatrixXd K;
    VectorXd k;
    const int nrows = args.grid_resolution.x();
    const int ncols = args.grid_resolution.y();
    compute_matrices ( args, K, k );

    MatrixXd Q = ( K.transpose() * K );
    VectorXd q = -2.0 * k.transpose() * K;
    int N = ncols + nrows;
    int N_iq = N;
    int N_eq = 2;

    // Equality constraint Ax = b
    MatrixXd A = MatrixXd::Zero ( N_eq, N );
    A.block ( 0, 0, 1, nrows ) = RowVectorXd::Ones ( nrows );
    A.block ( 1, nrows, 1, ncols ) = RowVectorXd::Ones ( ncols );
    VectorXd b ( N_eq );
    b ( 0 ) = args.dst_resolution.x();
    b ( 1 ) = args.dst_resolution.y();

    // Inequality constraints Gx <= h
    MatrixXd G = -MatrixXd::Identity ( N_iq, N );
    VectorXd h = VectorXd::Constant ( N_iq, -args.limits.x() );

    // Initial Solution vector X = [x s z y]
    // x = 0 s = 1 z = 1 y = 0
    VectorXd X = VectorXd::Zero ( N + N_iq + N_iq + N_eq );
    X.block ( N, 0, N_iq, 1 ) = VectorXd::Ones ( N_iq );
    X.block ( N + N_iq, 0, N_iq, 1 ) = VectorXd::Ones ( N_iq );

    // Setting up left hand side matrix ( only static parts )
    int N_LH = N + N_iq + N_iq + N_eq;
    MatrixXd LH = MatrixXd::Zero ( N_LH, N_LH );
    LH.block ( 0, 0, N, N ) = Q;
    LH.block ( N + N_iq, 0, N_iq, N ) = G;
    LH.block ( N + N_iq + N_iq, 0, N_eq, N ) = A;
    LH.block ( N + N_iq, N, N_iq, N ) = MatrixXd::Identity ( N_iq, N );
    LH.block ( 0, N + N_iq, N_iq, N ) = G.transpose();
    LH.block ( 0, N + N_iq + N_iq, N, N_eq ) = A.transpose();

    // Unpacks augmented solution vector
    auto unpack = [N, N_iq, N_eq] ( const MatrixXd & m, VectorXd & x, VectorXd & s, VectorXd & z, VectorXd & y ) {
        x = m.block ( 0, 0, N, 1 );
        s = m.block ( N, 0, N_iq, 1 );
        z = m.block ( N + N_iq, 0, N_iq, 1 );
        y = m.block ( N + N_iq + N_iq, 0, N_eq, 1 );
    };

    int iter = 0;
    double err = std::numeric_limits<double>::max();
    const double err_threshold = 1e-10;

    while ( err > err_threshold ) {

        VectorXd x, s, z, y;
        unpack ( X, x, s, z, y );

        // Updating dynamic block of LH
        MatrixXd Z = z.asDiagonal();
        MatrixXd S = s.asDiagonal();
        LH.block ( N, N, N_iq, N ) = Z;
        LH.block ( N, N + N_iq, N_iq, N ) = S;

        // Setting up Affine Step RH
        VectorXd RH_aff = VectorXd::Zero ( X.rows() );
        RH_aff.block ( 0, 0, N, 1 ) = - ( A.transpose() * y + G.transpose() * z + Q * x + q );
        RH_aff.block ( N, 0, N_iq, 1 ) = -S * z;
        RH_aff.block ( N + N_iq, 0, N_iq, 1 ) = - ( G * x + s - h );
        RH_aff.block ( N + N_iq + N_iq, 0, N_eq, 1 ) = - ( A * x - b );

        // Solving Affine Step
        VectorXd dX_aff = LH.fullPivLu().solve ( RH_aff );
        VectorXd x_aff, s_aff, z_aff, y_aff;
        unpack ( dX_aff, x_aff, s_aff, z_aff, y_aff );

        // computing mu, sigma and alpha_cc
        double sz = s.transpose() * z;
        double mu = sz / N_iq;
        double min_alpha = 0;

        for ( int i = 0; i < N_iq; ++i )
            if ( s_aff ( i ) < min_alpha * s ( i ) ) {
                min_alpha = s_aff ( i ) / s ( i );
            }

        for ( int i = 0; i < N_iq; ++i )
            if ( z_aff ( i ) < min_alpha * z ( i ) ) {
                min_alpha = z_aff ( i ) / z ( i );
            }

        double alpha;

        if ( -1 < min_alpha ) {
            alpha = 1;
        } else {
            alpha = -1 / min_alpha;
        }

        double sigma_num = ( ( s + alpha * s_aff ).transpose() * ( z + alpha * z_aff ) );
        double sigma = ( sigma_num / sz );
        sigma = sigma * sigma * sigma;

        // Computing Centering-Corrector Step RH
        VectorXd RH_cc = VectorXd::Zero ( X.rows() );
        RH_cc.block ( N, 0, N_iq, 1 ) = VectorXd::Constant ( N_iq, sigma * mu ) - s_aff.asDiagonal() * z_aff;

        // Solving Centering-Corrector Step
        VectorXd dX_cc = LH.fullPivLu().solve ( RH_cc );
        VectorXd dX = dX_aff + dX_cc;
        VectorXd dx, ds, dz, dy;
        unpack ( dX, dx, ds, dz, dy );
        min_alpha = 0;

        for ( int i = 0; i < N_iq; ++i )
            if ( ds ( i ) < min_alpha * s ( i ) ) {
                min_alpha = ds ( i ) / s ( i );
            }

        for ( int i = 0; i < N_iq; ++i )
            if ( dz ( i ) < min_alpha * z ( i ) ) {
                min_alpha = dz ( i ) / z ( i );
            }

        if ( -0.99 < min_alpha ) {
            alpha = 1;
        } else {
            alpha = -0.99 / min_alpha;
        }

        // Stepping
        X = X + alpha * dX;
        unpack ( X, x, s, z, y );

        // Evaluating erropr
        double gap = 0.0;

        for ( int i = 0; i < N_iq; ++i ) {
            gap += z ( i ) * s ( i );
        }

        double res1 = ( -A * x + b ).norm();
        err = ::std::max ( gap, res1 );
        ++iter;
    }

    if ( iters ) {
        *iters = iter;
    }

    solution = X.block ( 0, 0, N, 1 );
    return solution.minCoeff() > 0;
}
#endif

// Below here there is some more solvers code (will eventually remove it)
// - Penalty method (takes less iterations than interior point, but less nice convergence )
// - Equality-only 'iterative' solver which calculates the step solving for the KKT conditions
// ------------------------------------------------------------------------------------
#if 0
// Equality and Inequality constraints are transformed into penalty functions and we
// find the minimum iteratively following the gradient.
bool retarget_minimize_asap ( const RetargetArgs& args, MatrixXd& solution, int& iters ) {
    MatrixXd K;
    VectorXd k;
    const int nrows = args.grid_resolution.x();
    const int ncols = args.grid_resolution.y();
    compute_matrices ( args, K, k );
    // Inputs
    MatrixXd G = ( K.transpose() * K );
    VectorXd c = -2.0 * k.transpose() * K;
    int N = ncols + nrows;
    int N_iq = N;
    int N_eq = 2;
    export_matrix ( "G", G );
    export_matrix ( "c", c );
    // --- Setting up the matrices ---
    // Equality constraints guarantee the requested target resolution
    MatrixXd A = MatrixXd::Zero ( N_eq, N );
    A.block ( 0, 0, 1, nrows ) = RowVectorXd::Ones ( nrows );
    A.block ( 1, nrows, 1, ncols ) = RowVectorXd::Ones ( ncols );
    VectorXd b ( N_eq );
    b ( 0 ) = args.dst_resolution.x();
    b ( 1 ) = args.dst_resolution.y();
    // Initial solution
    VectorXd X = VectorXd::Zero ( N );

    for ( int i = 0; i < N; ++i ) {
        X ( i ) = -10.0;
    }

    double err = std::numeric_limits<double>::max();
    int iter = 0;
    double max_beta = 1e10;
    double max_gamma = 1e10;
    double beta = 10;
    double gamma = 10;

    while ( err > 1e-12 ) {
        MatrixXd Ai = MatrixXd::Identity ( N, N );

        for ( int i = 0; i < N; ++i ) {
            double limit = i <= ncols ? args.limits.x() : args.limits.y();

            if ( X ( i ) >= 0.15 ) {
                Ai ( i, i ) = 0.0;
            }
        }

        VectorXd bi = VectorXd::Constant ( N, 0.15 );
        beta = beta * 10;
        gamma = gamma * 10;
#if 0 // Direct solution 
        MatrixXd LH1 = G + beta * ( A.transpose() * A ) + gamma * Ai;
        VectorXd RH1 = c + beta * A.transpose() * b;
#else // Step
        MatrixXd LH1 = G + beta * ( A.transpose() * A ) + gamma * ( Ai );
        MatrixXd RH1 = c + beta * A.transpose() * b + gamma * Ai.transpose() * bi - ( G * X + beta * ( A.transpose() * A ) * X + gamma * ( Ai ) * X );
#endif
        VectorXd res1 = LH1.fullPivLu().solve ( RH1 );
        X = X + res1;
        // Equality constraint error
        err = res1.norm();
        bool ineq_violated = false;

        for ( int i = 0; i < N; ++i ) {
            if ( X ( i ) < args.limits.x() - 1e-6 ) {
                ineq_violated = true;
            }
        }

        cout << "Iteration " << iter << "\n" <<
             "Error (step norm) " << err << "\n" <<
             //"Inequalities violated " << (ineq_violated? "Yes" : "No") << endl;
             ++iter;
    }

    double w = 0;
    double h = 0;

    for ( int i = 0; i < N; ++i ) {
        if ( i < ncols ) {
            w += X ( i );
        } else {
            h += X ( i );
        }
    }

    iters = iter;
    solution = X;
    return true;
}
#endif

#if 0
// Inequality constraints are ignored here.
// The quadratic equality constrained problem is solved by solving the KKT system.
bool retarget_minimize_asap ( const RetargetArgs& args, MatrixXd& solution, int& iters ) {
    MatrixXd K;
    VectorXd k;
    const int nrows = args.grid_resolution.x();
    const int ncols = args.grid_resolution.y();
    compute_matrices ( args, K, k );
    // Inputs
    MatrixXd G = ( K.transpose() * K );
    VectorXd c = -2.0 * k.transpose() * K;
    int N = ncols + nrows;
    int N_iq = N;
    int N_eq = 2;
    export_matrix ( "G", G );
    export_matrix ( "c", c );
    // --- Setting up the matrices ---
    // Equality constraints guarantee the requested target resolution
    MatrixXd A = MatrixXd::Zero ( N_eq, N );
    A.block ( 0, 0, 1, nrows ) = RowVectorXd::Ones ( nrows );
    A.block ( 1, nrows, 1, ncols ) = RowVectorXd::Ones ( ncols );
    VectorXd b ( N_eq );
    b ( 0 ) = args.dst_resolution.x();
    b ( 1 ) = args.dst_resolution.y();
    // Initial solution
    // x
    VectorXd X = VectorXd::Zero ( N + N_eq );
    //for (int i = 0; i < ncols; ++i)
    //  X(i) = args.dst_resolution.x() / ncols;
    //for (int i = 0; i < nrows; ++i)
    //  X(ncols + i) = args.dst_resolution.y() / nrows;

    // y
    for ( int i = 0; i < N_eq; ++i ) {
        X ( N + i ) = 1.0;
    }

    MatrixXd LH = MatrixXd::Zero ( N + N_eq, N + N_eq );
    LH.block ( 0, 0, N, N ) = G;
    LH.block ( N, 0, N_eq, N ) = A;
    LH.block ( 0, N, N, N_eq ) = A.transpose();
    export_matrix ( "LH", LH );
    auto fname = [] ( const std::string & name, int idx ) -> std::string {
        return std::string ( "iter" ) + std::to_string ( idx ) + "-" + name;
    };
    double err = std::numeric_limits<double>::max();
    const double err_th = 1e-10;
    int iter = 0;

    while ( err > err_th ) {
        VectorXd x = X.block ( 0, 0, N, 1 );
        VectorXd h = A * x - b;
        VectorXd g = c + G * x;
        VectorXd RH ( N + N_eq );
        RH << g, h;
        //RH << c, b;
        export_matrix ( fname ( "RH", iter ), RH );
        VectorXd P = LH.fullPivLu().solve ( RH );
        VectorXd p = -P.block ( 0, 0, N, 1 );
        X.block ( 0, 0, N, 1 ) = x + p;
        double eq_residual = ( A * X.block ( 0, 0, N, 1 ) - b ).norm();
        double step_norm = p.norm();
        cout << "Iteration: " << iter << endl;
        cout << "Error (Residual): " << eq_residual << endl;
        cout << "Error (Step norm): " << step_norm << endl;
        err = ::std::max ( eq_residual, step_norm );
        ++iter;
    }

    iters = iter;
    solution = X.block ( 0, 0, N, 1 );
    return solution.minCoeff() >= 0;
}
#endif