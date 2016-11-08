/*
    Two utils where used:
    base64 => binary (https://github.com/danguer/blog-examples/blob/master/js/base64-binary.js) ( License Ok apparently, could also be rewritten )
    binary => double / u16... (http://jsfromhell.com/classes/binary-parser) ( Check license )
*/
// https://github.com/danguer/blog-examples/blob/master/js/base64-binary.js
var Base64Binary = 
{
    _keyStr : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=",
    
    /* will return a  Uint8Array type */
    decodeArrayBuffer: function(input) 
    {
        var bytes = (input.length/4) * 3;
        var ab = new ArrayBuffer(bytes);
        this.decode(input, ab);
        
        return ab;
    },

    removePaddingChars: function(input)
    {
        var lkey = this._keyStr.indexOf(input.charAt(input.length - 1));
        if(lkey == 64){
            return input.substring(0,input.length - 1);
        }
        return input;
    },

    decode: function (input, arrayBuffer) 
    {
        //get last chars to see if are valid
        input = this.removePaddingChars(input);
        input = this.removePaddingChars(input);

        var bytes = parseInt((input.length / 4) * 3, 10);
        
        var uarray;
        var chr1, chr2, chr3;
        var enc1, enc2, enc3, enc4;
        var i = 0;
        var j = 0;
        
        if (arrayBuffer)
            uarray = new Uint8Array(arrayBuffer);
        else
            uarray = new Uint8Array(bytes);
        
        input = input.replace(/[^A-Za-z0-9\+\/\=]/g, "");
        
        for (i=0; i<bytes; i+=3) {  
            //get the 3 octects in 4 ascii chars
            enc1 = this._keyStr.indexOf(input.charAt(j++));
            enc2 = this._keyStr.indexOf(input.charAt(j++));
            enc3 = this._keyStr.indexOf(input.charAt(j++));
            enc4 = this._keyStr.indexOf(input.charAt(j++));
    
            chr1 = (enc1 << 2) | (enc2 >> 4);
            chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
            chr3 = ((enc3 & 3) << 6) | enc4;
    
            uarray[i] = chr1;           
            if (enc3 != 64) uarray[i+1] = chr2;
            if (enc4 != 64) uarray[i+2] = chr3;
        }
    
        return uarray;  
    }
}

//+ Jonas Raoni Soares Silva
//@ http://jsfromhell.com/classes/binary-parser [rev. #1]
BinaryParser = function(bigEndian, allowExceptions)
{
    this.bigEndian = bigEndian, this.allowExceptions = allowExceptions;
};
with({p: BinaryParser.prototype}){
    p.decodeFloat = function(data, precisionBits, exponentBits){
        var b = ((b = new this.Buffer(this.bigEndian, data)).checkBuffer(precisionBits + exponentBits + 1), b),
            bias = Math.pow(2, exponentBits - 1) - 1, signal = b.readBits(precisionBits + exponentBits, 1),
            exponent = b.readBits(precisionBits, exponentBits), significand = 0,
            divisor = 2, curByte = b.buffer.length + (-precisionBits >> 3) - 1,
            byteValue, startBit, mask;
        do
            for(byteValue = b.buffer[ ++curByte ], startBit = precisionBits % 8 || 8, mask = 1 << startBit;
                mask >>= 1; (byteValue & mask) && (significand += 1 / divisor), divisor *= 2);
        while(precisionBits -= startBit);
        return exponent == (bias << 1) + 1 ? significand ? NaN : signal ? -Infinity : +Infinity
            : (1 + signal * -2) * (exponent || significand ? !exponent ? Math.pow(2, -bias + 1) * significand
            : Math.pow(2, exponent - bias) * (1 + significand) : 0);
    };
    p.decodeInt = function(data, bits, signed){
        var b = new this.Buffer(this.bigEndian, data), x = b.readBits(0, bits), max = Math.pow(2, bits);
        return signed && x >= max / 2 ? x - max : x;
    };
    with({p: (p.Buffer = function(bigEndian, buffer){
        this.bigEndian = bigEndian || 0, this.buffer = [], this.setBuffer(buffer);
    }).prototype}){
        p.readBits = function(start, length){
            //shl fix: Henri Torgemane ~1996 (compressed by Jonas Raoni)
            function shl(a, b){
                for(++b; --b; a = ((a %= 0x7fffffff + 1) & 0x40000000) == 0x40000000 ? a * 2 : (a - 0x40000000) * 2 + 0x7fffffff + 1);
                return a;
            }
            if(start < 0 || length <= 0)
                return 0;
            this.checkBuffer(start + length);
            for(var offsetLeft, offsetRight = start % 8, curByte = this.buffer.length - (start >> 3) - 1,
                lastByte = this.buffer.length + (-(start + length) >> 3), diff = curByte - lastByte,
                sum = ((this.buffer[ curByte ] >> offsetRight) & ((1 << (diff ? 8 - offsetRight : length)) - 1))
                + (diff && (offsetLeft = (start + length) % 8) ? (this.buffer[ lastByte++ ] & ((1 << offsetLeft) - 1))
                << (diff-- << 3) - offsetRight : 0); diff; sum += shl(this.buffer[ lastByte++ ], (diff-- << 3) - offsetRight)
            );
            return sum;
        };
        p.setBuffer = function(data){
            if(data){
                for(var l, i = l = data.length, b = this.buffer = new Array(l); i; b[l - i] = data.charCodeAt(--i));
                this.bigEndian && b.reverse();
            }
        };
        p.hasNeededBits = function(neededBits){
            return this.buffer.length >= -(-neededBits >> 3);
        };
        p.checkBuffer = function(neededBits){
            if(!this.hasNeededBits(neededBits))
                throw new Error("checkBuffer::missing bytes");
        };
    }
    p.warn = function(msg){
        if(this.allowExceptions)
            throw new Error(msg);
        return 1;
    };
    p.toShort = function(data){return this.decodeInt(data, 16, true);};
    p.toDouble = function(data){return this.decodeFloat(data, 52, 11);};
}

var Utils =
{
    base64_to_double : function(base64_str)
    {
        // base64 => arraybuffer
        var array_buffer = Base64Binary.decodeArrayBuffer(base64_str);

        // arraybuffer => binary str
        var binary_str = String.fromCharCode.apply(null, new Uint8Array(array_buffer));

        // binary str => double
        var binary_parser = new BinaryParser();
        return binary_parser.toDouble(binary_str);
    },

    base64_to_u16 : function(base64_str)
    {
        // base64 => arraybuffer
        var array_buffer = Base64Binary.decodeArrayBuffer(base64_str);

        // arraybuffer => binary str
        var binary_str = String.fromCharCode.apply(null, new Uint8Array(array_buffer));

        // binary str => u16
        var binary_parser = new BinaryParser();
        return binary_parser.toShort(binary_str);
    }
}

var RetargetGL = 
{
    id_counter : 0,
    rcs : [],
    setup_element : function(element, metadata, callback)
    {   
        var rc = { } 
        var canvas = element;
        rc.gl = canvas.getContext("experimental-webgl");

        /*
            Creating shaders
        */
        var vs_code = `
            attribute vec3 position_in;
            attribute vec2 texcoord_in; 

            varying vec2 texcoord_out;

            void main()
            {
                gl_Position = vec4(position_in, 1.0);
                texcoord_out = texcoord_in;
            }
        `;

        var ps_code = `
            precision mediump float;
            varying vec2 texcoord_out;
            uniform sampler2D default_sampler;

            void main() 
            {
               gl_FragColor = texture2D(default_sampler, vec2(texcoord_out.s, texcoord_out.t));
            }
        `;

        var grid_ps_code = `
            precision mediump float;
            varying vec2 texcoord_out;

            void main() 
            {
               gl_FragColor = vec4(0, 0, 0, 1.0);
            }
        `;

        var vs = rc.gl.createShader(rc.gl.VERTEX_SHADER);
        rc.gl.shaderSource(vs, vs_code);
        rc.gl.compileShader(vs);

        if (!rc.gl.getShaderParameter(vs, rc.gl.COMPILE_STATUS))
            Retarget.log("Failed to compile shader: " + rc.gl.getShaderInfoLog(vs));

        var ps = rc.gl.createShader(rc.gl.FRAGMENT_SHADER);
        rc.gl.shaderSource(ps, ps_code);
        rc.gl.compileShader(ps);

        if (!rc.gl.getShaderParameter(ps, rc.gl.COMPILE_STATUS))
            Retarget.log("Failed to compile shader: " + rc.gl.getShaderInfoLog(ps));        

        var grid_ps = rc.gl.createShader(rc.gl.FRAGMENT_SHADER);
        rc.gl.shaderSource(grid_ps, grid_ps_code);
        rc.gl.compileShader(grid_ps);

        if (!rc.gl.getShaderParameter(grid_ps, rc.gl.COMPILE_STATUS))
            Retarget.log("Failed to compile shader: " + rc.gl.getShaderInfoLog(grid_ps));

        rc.program = rc.gl.createProgram();
        rc.gl.attachShader(rc.program, vs);
        rc.gl.attachShader(rc.program, ps);
        rc.gl.linkProgram(rc.program);

        rc.grid_program = rc.gl.createProgram();
        rc.gl.attachShader(rc.grid_program, vs);
        rc.gl.attachShader(rc.grid_program, grid_ps);
        rc.gl.linkProgram(rc.grid_program);

        if (!rc.gl.getProgramParameter(rc.program, rc.gl.LINK_STATUS))
            Retarget.log("Failed to link program");

        // getting attribs locations
        rc.program.position_in = rc.gl.getAttribLocation(rc.program, 'position_in');
        rc.gl.enableVertexAttribArray(rc.program.position_in);

        rc.program.texcoord_in = rc.gl.getAttribLocation(rc.program, 'texcoord_in');
        rc.gl.enableVertexAttribArray(rc.program.texcoord_in);

        /*
            Creating buffer
        */
        // Uvs are fixed ( as cropping is not dynamic ) and indices are static
        // too, Positions are the only ones that are dynamic. UVs are currently 
        // duplicated ( TODO provide them as uniform buffer )
        var row_num = metadata.cells_y + 1;
        var col_num = metadata.cells_x + 1;

        rc.positions = new Float32Array(row_num * col_num * 3);
        var uvs = new Float32Array(row_num * col_num * 2);
        var indices = new Uint16Array(metadata.cells_x * metadata.cells_y * 2 * 3);

        // UV
        for (var y = 0; y < row_num; ++y)
        {
            for (var x = 0; x < col_num; ++x)
            {
                uvs[2 * (y * col_num + x)] = x / (col_num - 1);
                uvs[2 * (y * col_num + x) + 1] = y / (row_num - 1);
            }
        }

        // Indices
        for (var y = 0; y < metadata.cells_y; ++y)
        {
            for (var x = 0; x < metadata.cells_x; ++x)
            {
                var v0 = y * col_num + x;   // TL
                var v1 = v0 + 1;            // TR
                var v3 = v0 + col_num;      // BR
                var v2 = v3 + 1;            // BL
            
                var idx = 6 * (y * metadata.cells_x + x);

                // Triangle 1
                indices[idx]     = v1;
                indices[idx + 1] = v0;
                indices[idx + 2] = v3;

                // Triangle 2
                indices[idx + 3] = v3;
                indices[idx + 4] = v2;
                indices[idx + 5] = v1;
            }
        }

        // Positions 
        rc.pos_vbo = rc.gl.createBuffer();
        rc.gl.bindBuffer(rc.gl.ARRAY_BUFFER, rc.pos_vbo);
        //rc.gl.bufferData(rc.gl.ARRAY_BUFFER, rc.positions, rc.gl.DYNAMIC_DRAW);
        rc.pos_vbo.num_components = 3;
        
        // Uvs
        rc.tex_vbo = rc.gl.createBuffer();
        rc.gl.bindBuffer(rc.gl.ARRAY_BUFFER, rc.tex_vbo); 
        rc.gl.bufferData(rc.gl.ARRAY_BUFFER, uvs, rc.gl.STATIC_DRAW);
        rc.tex_vbo.num_components = 2;

        // Indices
        rc.ibo = rc.gl.createBuffer();
        rc.gl.bindBuffer(rc.gl.ELEMENT_ARRAY_BUFFER, rc.ibo);
        rc.gl.bufferData(rc.gl.ELEMENT_ARRAY_BUFFER, indices, rc.gl.STATIC_DRAW);
        rc.ibo.num_elements = indices.length;
        /////////////////////////////////////////////////

        /*
            Creating texture
        */
        rc.texture = rc.gl.createTexture();
        rc.texture_image = new Image()

        rc.texture_image.onload = function()
        {
            rc.gl.bindTexture(rc.gl.TEXTURE_2D, rc.texture);
            rc.gl.texImage2D(rc.gl.TEXTURE_2D, 0, rc.gl.RGBA, rc.gl.RGBA, rc.gl.UNSIGNED_BYTE, rc.texture_image);
            rc.gl.texParameteri(rc.gl.TEXTURE_2D, rc.gl.TEXTURE_MAG_FILTER, rc.gl.LINEAR);

            // Required to use non power of two textures
            // https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_textures_in_WebGL
            rc.gl.texParameteri(rc.gl.TEXTURE_2D, rc.gl.TEXTURE_MIN_FILTER, rc.gl.LINEAR);
            rc.gl.texParameteri(rc.gl.TEXTURE_2D, rc.gl.TEXTURE_WRAP_S, rc.gl.CLAMP_TO_EDGE);
            rc.gl.texParameteri(rc.gl.TEXTURE_2D, rc.gl.TEXTURE_WRAP_T, rc.gl.CLAMP_TO_EDGE);
            rc.gl.bindTexture(rc.gl.TEXTURE_2D, null);
            
            //rc.texture.id = RetargetGL.id_counter++;
            // Notifying caller that everything went as expected 
            callback(element);

            RetargetGL.rcs.push(rc);
            window.addEventListener('resize', function() 
            {
                RetargetGL.render_callback(rc);
            });

            RetargetGL.render_callback(rc);
        };

        rc.metadata = metadata;
        rc.texture_image.src = 'data:image/jpeg;base64, ' + metadata.raw_data;
   },

    bind_and_update_grid : function(rc)
    {
        var spacings = new Array(rc.metadata.cells_x + rc.metadata.cells_y);
        var cur_w = rc.gl.canvas.clientWidth;
        var cur_h = rc.gl.canvas.clientHeight;
        var cur_ar = Math.log(cur_w / cur_h);

        // Finding upper and minimum aspect ratios
        max_spacings = null;
        max_ar = Infinity;

        min_spacings = null;
        min_ar = -Infinity;

        // TODO : Binary search pls
        for (var i = 0; i < rc.metadata.sorted_ars.length; ++i)
        {
            var [w, h] = Retarget.name_to_resolution(rc.metadata.sorted_ars[i]);
            var ar = Math.log(w/h);

            if (ar > cur_ar && ar < max_ar)
            {
                max_ar = ar;
                max_spacings = rc.metadata.spacings[rc.metadata.sorted_ars[i]];
            }
            else if (ar < cur_ar && ar > min_ar)
            {
                min_ar = ar;
                min_spacings = rc.metadata.spacings[rc.metadata.sorted_ars[i]];
            }
        }

        // Handling out of range aspect ratios
        // Just using closest one ( either max or min )
        if (min_spacings == null && max_spacings == null)
        {
            Retarget.log("Failed to find matching aspect ratios"); 
        }
        else if (min_spacings == null)
        {
            min_spacings = max_spacings;
            min_ar = max_ar;
        }
        else if (max_spacings == null)
        {
            max_spacings = min_spacings;
            max_ar = min_ar;
        }

        // Interpolating values 
        // https://www.cmu.edu/biolphys/deserno/pdf/log_interpol.pdf
        function log_interp(a, b, factor)
        {
            return Math.pow(b, factor) * Math.pow(a, 1 - factor); // a + (b - a) * factor;
        }        

        
        // Handling out-of-range ARs
        var log_interp_factor = (cur_ar - min_ar) / (max_ar - min_ar);
        if (log_interp_factor == Infinity)
            log_interp_factor = 1.0;
        if (log_interp_factor == -Infinity)
            log_interp_factor = 1.0;

        // TODO: Can i predict final width ?
        var total_width = 0;
        for (var i = 0; i < rc.metadata.cells_x; ++i)
        {
            spacings[i] = log_interp(min_spacings[i], max_spacings[i], log_interp_factor);
            total_width += spacings[i];
        }

        var total_height = 0;
        for (var i = rc.metadata.cells_x; i < rc.metadata.cells_x + rc.metadata.cells_y; ++i)
        {
            spacings[i] = log_interp(min_spacings[i], max_spacings[i], log_interp_factor);
            total_height += spacings[i];
        }

        // Scaling
        var scale_factor_x = 2 / total_width;
        var scale_factor_y = 2 / total_height;

        for (var i = 0; i < rc.metadata.cells_x; ++i)
        {
            spacings[i] *= scale_factor_x;
        }

        for (var i = rc.metadata.cells_x; i < rc.metadata.cells_x + rc.metadata.cells_y; ++i)
        {
            spacings[i] *= scale_factor_y;
        }

        var row_num = rc.metadata.cells_y + 1;
        var col_num = rc.metadata.cells_x + 1;

        // Updating grid with new offsets
        for (var y = 0; y < row_num; ++y)
        {
            for (var x = 0; x < col_num; ++x)
            {
                var idx = 3 * (y * col_num + x);

                off_x = spacings[x - 1];
                off_y = spacings[rc.metadata.cells_x + y - 1];

                var prev_x_idx = idx - 3;
                var prev_y_idx = (idx + 1) - col_num * 3;

                // Offsets are 0 on edges, could reimplement adding 0 spacings ( TODO )
                // for cleaner code.
                if (x >= 1)
                    rc.positions[idx] = rc.positions[prev_x_idx] + off_x;
                else
                    rc.positions[idx] = - 1;

                if (y >= 1)
                    rc.positions[idx + 1] = rc.positions[prev_y_idx] - off_y;
                else
                    rc.positions[idx + 1] = 1;

                rc.positions[idx + 2] = 0;
            }
        }
        rc.gl.bindBuffer(rc.gl.ARRAY_BUFFER, rc.pos_vbo);
        rc.gl.bufferData(rc.gl.ARRAY_BUFFER, rc.positions, rc.gl.DYNAMIC_DRAW);
    },

    render_callback : function(rc)
    {
        rc.gl.canvas.width = rc.gl.canvas.clientWidth;
        rc.gl.canvas.height = rc.gl.canvas.clientHeight;

        rc.gl.viewport(0, 0, rc.gl.drawingBufferWidth, rc.gl.drawingBufferHeight);
        rc.gl.clearColor(0.15, 0.15, 0.15, 1);
        rc.gl.clear(rc.gl.COLOR_BUFFER_BIT | rc.gl.DEPTH_BUFFER_BIT);

        rc.gl.useProgram(rc.program);

        // positions
        RetargetGL.bind_and_update_grid(rc);

        //rc.gl.bindBuffer(rc.gl.ARRAY_BUFFER, rc.pos_vbo);
        rc.gl.vertexAttribPointer(rc.program.position_in, rc.pos_vbo.num_components, rc.gl.FLOAT, false, 0, 0);

        // Texcoord
        rc.gl.bindBuffer(rc.gl.ARRAY_BUFFER, rc.tex_vbo);
        rc.gl.vertexAttribPointer(rc.program.texcoord_in, rc.tex_vbo.num_components, rc.gl.FLOAT, false, 0, 0);

        // Index buffer
        rc.gl.bindBuffer(rc.gl.ELEMENT_ARRAY_BUFFER, rc.ibo);

        // Texture
        rc.gl.activeTexture(rc.gl.TEXTURE0);
        rc.gl.bindTexture(rc.gl.TEXTURE_2D, rc.texture);

        rc.gl.drawElements(rc.gl.TRIANGLES, rc.ibo.num_elements, rc.gl.UNSIGNED_SHORT, 0);

       // rc.gl.useProgram(rc.grid_program);
        //rc.gl.drawElements(rc.gl.LINES, rc.ibo.num_elements, rc.gl.UNSIGNED_SHORT, 0);
    }
}

var Retarget =  
{
    name_to_resolution : function(name)
    {
        var match = /RetargetSpacings([0-9]*)x([0-9]*)/g.exec(name);
        return [Number(match[1]), Number(match[2])];
    },

    check_features : function()
    {
        if (!XMLHttpRequest)
            return false;

        if (!window.atob)
            return false;

        if (!DOMParser)
            return false;

        return true;
    },

    logging_enabled : true,

    log : function(msg) 
    {
        if (Retarget.logging_enabled)
            console.log(msg);
    },

    extract_metadata_xmp : function(img, callback)
    {
        function extract_spacings(element)
        {
            if (element == undefined)
                return null;

            if (!element.hasChildNodes())
                return null;

            var seq_element = null;
            for (var i = 0; i < element.childNodes.length; ++i)
            {
                if (element.childNodes[i].tagName == "rdf:Seq")
                {
                    seq_element = element.childNodes[i];
                    break;
                }
            }

            if (seq_element == null)
                return null;

            spacings = [];
            for (var i = 0; i < seq_element.childNodes.length; ++i)
            {
                var li_element = seq_element.childNodes[i];
                if (li_element.tagName == "rdf:li")
                {
                    spacings.push(Utils.base64_to_double(li_element.innerHTML));
                }

            }

            return spacings;
        }

        function extract_callback(ascii_str)
        {
            var str = ascii_str;

            var ret = { }
            ret.spacings = { } 

            while (true)
            {
                // Finding start/end xmp tags, slicing the buffer and feeding it
                // to a DOMParser()
                var start_xmp_idx = str.indexOf("<?xpacket begin=");

                // This finds the begging of the end tag, just moving till the end
                var end_xmp_idx = str.indexOf("<?xpacket end=");
                end_xmp_idx += str.substring(end_xmp_idx).indexOf(">") + 1;

                if (start_xmp_idx == -1 || end_xmp_idx == -1)
                {
                    // if no packet is found data could be contained in the extended section. 
                    // as of 2.1.3.1 [http://www.adobe.com/content/dam/Adobe/en/devnet/xmp/pdfs/XMPSpecificationPart3.pdf]
                    // data is not a serialized inside a packet wrapper, thus we look for APP1 marker segments 
                    // <x:xmpmeta </x:xmpmeta>
                    start_xmp_idx = str.indexOf("<x:xmpmeta");
                    end_xmp_idx = str.indexOf("</x:xmpmeta>");
                    end_xmp_idx += str.substring(end_xmp_idx).indexOf(">") + 1;

                    if (start_xmp_idx == -1 || end_xmp_idx == -1)
                        break;
                }

                Retarget.log("Found xmp start: " + start_xmp_idx + " for: " + img.src);
                Retarget.log("Found xmp end: " + end_xmp_idx + " for: " + img.src);

                xmp_data = str.substring(start_xmp_idx, end_xmp_idx);

                // Thanks a lot
                // http://stackoverflow.com/questions/14665288/removing-invalid-characters-from-xml-before-serializing-it-with-xmlserializer
                // TODO: Properly extend based on specification if needed
                var NOT_SAFE_IN_XML_1_0 = /[^\x09\x0A\x0D\x20-\xFF\x85\xA0-\uD7FF\uE000-\uFDCF\uFDE0-\uFFFD]/gm;
                xmp_data = xmp_data.replace(NOT_SAFE_IN_XML_1_0, '');

                var parser = new DOMParser();
                var doc = parser.parseFromString(xmp_data, "text/xml");

                var parser_error = doc.getElementsByTagName('parsererror')[0];
                if (parser_error != undefined)
                {
                    Retarget.log("Parsing not complete with error: " + parser_error.getElementsByTagName('div')[0].innerHTML)
                }

                // xmp NS = http://ns.adobe.com/xap/1.0/ 
                function doc_find(name) { return doc.getElementsByTagNameNS("http://ns.adobe.com/xap/1.0/", name)[0]; };

                // Looking for spacings
                desc_elements = doc.getElementsByTagNameNS("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "Description");
                for (var de = 0; de < desc_elements.length; ++de)
                {
                    var desc_element = desc_elements[de];
                    var cells_x_attr = desc_element.getAttributeNS("http://ns.adobe.com/xap/1.0/", "RetargetCellsX");
                    if (cells_x_attr != undefined)
                        ret.cells_x = Utils.base64_to_u16(cells_x_attr);
                    
                    var cells_y_attr = desc_element.getAttributeNS("http://ns.adobe.com/xap/1.0/", "RetargetCellsY");
                    if (cells_y_attr != undefined)
                        ret.cells_y = Utils.base64_to_u16(cells_y_attr);

                    for (var c = 0; c < desc_element.childNodes.length; ++c)
                    {
                        var child = desc_element.childNodes[c];
                        if (/^xmp:RetargetSpacings[0-9]+\x[0-9]+/.test(child.tagName))
                        {
                            ret.spacings[child.tagName] = extract_spacings(child);
                        }
                    }
                }

                str = str.substring(end_xmp_idx);
            }

            // TODO: Remove duplicates
            ret.sorted_ars = []
            for (var name in ret.spacings)
            {
                ret.sorted_ars.push(name);
            }

            ret.sorted_ars.sort(function(a, b)
            {
                var [w1, h1] = Retarget.name_to_resolution(a);
                var [w2, h2] = Retarget.name_to_resolution(b);

                return w2/h2 - w1/h1;
            });

            return ret;
        }

        Retarget.log("Extracting metadata for: " + img.src);

        // Currently assuming img.src is **not** a data URI, but the url 
        if (/^data\:/.test(img.src))
        {
            Retarget.log("Found data for: " + img);
            callback(img, extract_callack(window.atob(img.src)));    
        }
        else
        {
            Retarget.log("Requesting: " + img.src);
            var req = new XMLHttpRequest();
            req.open('GET', img.src, true);
            req.responseType = 'arraybuffer';
            req.addEventListener("load", function()
            {
                Retarget.log("Received raw data for: " + img.src);

                /* TextDecoder is not supported in safari yet
                var data_view = new DataView(this.response);
                var decoder = new TextDecoder('ascii');
                var buf_as_str = decoder.decode(data_view); */
                // appending raw data used to load original image
                // TODO: Can this be improved ? 
                var buf_as_str = '';
                var bytes = new Uint8Array(this.response);
                var len = bytes.byteLength;
                for (var i = 0; i < len; ++i)
                {
                    buf_as_str += String.fromCharCode(bytes[i]);
                }
                metadata = extract_callback(buf_as_str);

                metadata.raw_data = window.btoa(buf_as_str);
                //console.log(metadata.raw_data.length);
                // stack overflows for big buffers 
                // metadata.raw_data = window.btoa(String.fromCharCode.apply(null, new Uint8Array(this.response)));;

                callback(img, metadata);
            });

            req.send(null);
        }
    },

    replace : function(document, window, img, metadata)
    {
        Retarget.log("Received metadata for " + img.src)

        var new_canvas = document.createElement('canvas');

        // window.getComputedStyle() returns computed values, thus width/height/etc... are in pixels
        // the trick is to hide the image and then request the computed style ( that won't finish
        // computing the pixel values ), image will then be redisplayed right after computing the values.
        img.style.display = 'none';
        var style = window.getComputedStyle(img);
        
        //new_canvas.style.cssText = style.cssText;
        new_canvas.style.width = style.width;
        new_canvas.style.height = style.height;
        new_canvas.style.minWidth = style.minWidth;
        new_canvas.style.minHeight = style.minHeight;
        new_canvas.style.maxWidth = style.maxWidth;
        new_canvas.style.maxHeight = style.maxHeight;
        new_canvas.style.float = style.float;
        new_canvas.style.border = style.border;

        // Redisplaying the image
        img.style.display = 'block';

        RetargetGL.setup_element(new_canvas, metadata, function(element)
        {
            // Swapping
            img.style.display = 'none';
            new_canvas.style.display = 'block';
            img.parentNode.insertBefore(new_canvas, img);
        });
    },

    run_internal : function(document, window)
    {
        Retarget.log("DOM content has been loaded, finding all <img>...");
        var imgs = document.getElementsByTagName('img');

        // Hooking up resize event
        window.addEventListener('resize', Retarget.on_resize);

        for (var i = 0; i < imgs.length; ++i)
        {
            if (imgs[i].getAttribute("data-retarget") == undefined)
                continue;
   
            Retarget.log("Found: " + imgs[i].src);

            // Callback has to take the img otherwise i lose the reference ??
            Retarget.extract_metadata_xmp(imgs[i], function(img, metadata)
            {
                Retarget.replace(document, window, img, metadata);
            });
        }
    },

    run(document, window)
    {
        Retarget.log("Hooking up DOMContentLoaded event...");

        // We don't need to wait for load since we don't need to wait for all the resources
        // to be downloaded ( might need to request them again if not embedded in src ).
        // Just waiting for the DOM content is enough (WebGL ?)
        document.addEventListener("DOMContentLoaded", function(event) 
        {
            Retarget.run_internal(document, window);
        });
    },
}

// Run!
Retarget.run(document, window);
