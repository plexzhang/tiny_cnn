/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/activations/activation_function.h"
#include <deque>

namespace tiny_cnn {

struct connection_table {
    connection_table() : rows_(0), cols_(0) {}
    connection_table(const bool *ar, cnn_size_t rows, cnn_size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    }
    connection_table(cnn_size_t ngroups, cnn_size_t rows, cnn_size_t cols) : connected_(rows * cols, false), rows_(rows), cols_(cols) {
        if (rows % ngroups || cols % ngroups) throw nn_error("invalid group size");

        cnn_size_t row_group = rows / ngroups;
        cnn_size_t col_group = cols / ngroups;

        for (cnn_size_t g = 0; g < ngroups; g++) {
            for (cnn_size_t r = 0; r < row_group; r++)
              for (cnn_size_t c = 0; c < col_group; c++)
                connected_[(r + g * row_group) * cols_ + c + g * col_group] = true;
        }
    }

    bool is_connected(cnn_size_t x, cnn_size_t y) const {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() const {
        return rows_ == 0 && cols_ == 0;
    }

    std::deque<bool> connected_;
    cnn_size_t rows_;
    cnn_size_t cols_;
};

enum class padding {
    valid, ///< use valid pixels of input
    same   ///< add zero-padding around input so as to keep image size
};


template<typename Activation = activation::identity>        //默认情况下, Activation = activation::identity
class convolutional_layer : public layer<Activation> {          //template<typename Activation>  class layer : public layer_base {}
                                                                                                  //class identity: public function{ f(), df(), scale()}
public:
    typedef layer<Activation> Base;     //layer(in_dim, out_dim, weight_dim, bias_dim)
    CNN_USE_LAYER_MEMBERS;  //#define *** using layer_base  ....

    using layer_base::out_size;


    /*************第一种形式***************/
    /**
    * constructing convolutional layer
    *
    * @param in_width     [in] input image width
    * @param in_height    [in] input image height
    * @param window_size  [in] window(kernel) size of convolution
    * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels [in] output image channels
    * @param padding      [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1)
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_size, w_stride, h_stride, pad_type) * out_channels,
            sqr(window_size) * in_channels * out_channels, has_bias ? out_channels : 0),
        in_(in_width, in_height, in_channels),
        in_padded_( in_length(in_width, window_size, pad_type),   in_length(in_height, window_size, pad_type),  in_channels), 
        // in_padded_   表示补白之后的输入size，即(width, height, depth)      
        // padding_type==same,  padding_length=in_width+window_size-1           type==valid,    padding_length=in_width
        out_(conv_out_length(in_width, window_size, w_stride, pad_type), conv_out_length(in_height, window_size, h_stride, pad_type), out_channels),
        // out_     表示输出的size，(width, height, depth)   
        //  根据padding_type，same为in_width；   valid为(in_width-window_size+1)/stride
        weight_(window_size, window_size, in_channels*out_channels),
        // weight_      表示权值，window * window * in_channels * out_channels
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();         //init初始化，假设padding_type为valid，置prev_out_buf_[i] = nullptr;
    }       //构造函数，window_size


    /*************第二种形式***************/
    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels  [in] output image channels
    * @param padding       [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_width,
        cnn_size_t window_height,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1)
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_width, window_height, w_stride, h_stride, pad_type) * out_channels,
            window_width*window_height * in_channels * out_channels, has_bias ? out_channels : 0),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_width, pad_type), in_length(in_height, window_height, pad_type), in_channels),
        out_(conv_out_length(in_width, window_width, w_stride, pad_type), conv_out_length(in_height, window_height, h_stride, pad_type), out_channels),
        weight_(window_width, window_height, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }   //构造函数, window_width和window_height


    /*************第三种形式***************/
    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_size      [in] window(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        const connection_table& connection_table,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1
        )
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_size, w_stride, h_stride, pad_type) * out_channels,
            sqr(window_size) * in_channels * out_channels, has_bias ? out_channels : 0),
        tbl_(connection_table),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_size, pad_type), in_length(in_height, window_size, pad_type), in_channels),
        out_(conv_out_length(in_width, window_size, w_stride, pad_type), conv_out_length(in_height, window_size, h_stride, pad_type), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }       //构造函数, 使用了connection_table, window_size

    /*************第四种形式***************/
    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_width,
        cnn_size_t window_height,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        const connection_table& connection_table,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1
        )
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_width, window_height, w_stride, h_stride, pad_type) * out_channels,
            window_width*window_height * in_channels * out_channels, has_bias ? out_channels : 0),
        tbl_(connection_table),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_width, pad_type), in_length(in_height, window_height, pad_type), in_channels),
        out_(conv_out_length(in_width, window_width, w_stride, pad_type), conv_out_length(in_height, window_height, h_stride, pad_type), out_channels),
        weight_(window_width, window_height, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }     //构造函数, 使用了connection_table, window_width和window_height



    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const override
    {
        return weight_.width_ * weight_.height_ * in_.depth_;       //输入单元数
    }

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const override
    {
        return (weight_.width_ / w_stride_) * (weight_.height_ / h_stride_) * out_.depth_;      //输出单元数
    }

    ///< number of connections
    virtual size_t connection_size() const override
    {
        return out_.size() * fan_in_size();     //该层的连接数
    }



    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) override
    {
        const vec_t& prev_out = *(prev_out_padded_[0]);     
        const activation::function& prev_h = prev_->activation_function();
        vec_t* prev_delta = (pad_type_ == padding::same) ? &prev_delta2_padded_ : &prev_delta2_;

        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

        // accumulate dw
        for_i(in_.depth_, [&](int inc) {            // for_(parallelize, 0, size, [&](const blocked_range& r) {for (int i = r.begin(); i < r.end(); i++)}, grainsize);  0为begin, size为end
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
                        const float_t * delta = &current_delta2[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            for (cnn_size_t x = 0; x < out_.width_; x++) {
                                dst += sqr(prevo[y * in_padded_.width_ + x]) * delta[y * out_.width_ + x];
                            }
                        }
                        Whessian_[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (!this->bhessian_.empty()) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const float_t *delta = &current_delta2[out_.get_index(0, 0, outc)];
                this->bhessian_[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));
            }
        }

        // propagate delta to previous layer
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                if (!tbl_.is_connected(outc, inc)) continue;

                const float_t *pw = &W_[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
                const float_t *pdelta_src = &current_delta2[out_.get_index(0, 0, outc)];
                float_t *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];
                        float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[wy * in_padded_.width_ + wx] += sqr(*ppw++) * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        for_i(parallelize_, in_padded_.size(), [&](int i) {
            (*prev_delta)[i] *= sqr(prev_h.df(prev_out[i]));
        });

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(prev_delta2_padded_, prev_delta2_);

        CNN_LOG_VECTOR(current_delta2, "[pc]curr-delta2");
        CNN_LOG_VECTOR(prev_delta2_, "[pc]prev-delta2");
        CNN_LOG_VECTOR(Whessian_, "[pc]whessian");

        return prev_->back_propagation_2nd(prev_delta2_);
    }  //const vec_t& back_propagation_2nd(const vec_t& current_delta2)



    virtual const vec_t& forward_propagation(const vec_t& in_raw, size_t worker_index) override
    {
        copy_and_pad_input(in_raw, static_cast<int>(worker_index));         //prev_out_padded_[i] = in_raw

        vec_t &a = a_[worker_index];                                                              // a=w*x
        vec_t &out = output_[worker_index];                                                 // output
        const vec_t &in = *(prev_out_padded_[worker_index]);                   // input，      vec_t*  prev_out_padded_[CNN_TASK_SIZE]; 
        
        std::fill(a.begin(), a.end(), float_t(0));                                                  // 用0初始化a                               


        //out_.depth_, [&](int o)表示 for(o=0; o<out_.depth; o++)
        for_i(parallelize_, out_.depth_, [&](int o) {                     //for_i(bool parallelize, T size, Func f, int grainsize=100)
            for (cnn_size_t inc = 0; inc < in_.depth_; inc++) {             // { for_(parallelize, 0, size, [&](const blocked_range& r) { for(int i = r.begin(); i < r.end(); i++) {f(i) }}, g); }
                if (!tbl_.is_connected(o, inc)) continue;                                       //  is_connected()      返回   is_empty()? true : connection_[]                                   
                // in_.depth表示输出的channel总数， o表示第o个输出map

                const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * o + inc)];           //index3d -> get_index  { return (height_ * channel + y) * width_ + x}，即height*width*(in_.depth*o+inc)
                const float_t *pi = &in[in_padded_.get_index(0, 0, inc)];                                           // vec_t  W_；    表示weight的具体值；  vec_t in_padded;  in表示具体输入值
                float_t *pa = &a[out_.get_index(0, 0, o)];                                                  //pw表示point weight，pi表示point in，pa表示point a 都为具体的单数值
                                                                                                                                    //  第o个map的具体值，索引位置为 height*width*o

                for (cnn_size_t y = 0; y < out_.height_; y++) {                                         //根据输出out的size
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;                //第一次初始化，第二次将w复位
                        const float_t * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;     //当stride=1时，即ppi=pi+y*in_padded_.width+x
                        float_t sum = float_t(0);                                                                                             //通过指针变化，移动输入与权值

                        // should be optimized for small kernel(3x3,5x5)
                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                sum += *ppw++ * ppi[wy * in_padded_.width_ + wx];               //sum+=(*pw)*ppi[x]，pw++
                            }                                                                                                        //ppi为指针，则ppi[wy*in_padded_.width + wx]为指向值
                        }
                        pa[y * out_.width_ + x] += sum;                                                     //计算得到sum(w*w)
                    }
                }
            }

            if (!this->b_.empty()) {                                                                            //判断b非空
                float_t *pa = &a[out_.get_index(0, 0, o)];                                          //pa=&a，刚才修改的pa则返回到a中，即a已修改
                float_t b = this->b_[o];
                std::for_each(pa, pa + out_.width_ * out_.height_, [&](float_t& f) { f += b; });        //在pa到pa+out_.width*out_height的范围内，f=pa，执行pa+=b，以引用方式执行
            }
        });         //for_i(parallelize_, out_.depth_, [&](int o) { });


        for_i(parallelize_, out_size_, [&](int i) {                     //f为激励函数
            out[i] = h_.f(a, i);                                                    //Activation h_;    vec_t &out =  output_[worker_index];  
        });

        CNN_LOG_VECTOR(in_raw, "[pc]in");
        CNN_LOG_VECTOR(W_, "[pc]w");
        CNN_LOG_VECTOR(a, "[pc]a");
        CNN_LOG_VECTOR(out, "[pc]forward");

        return next_ ? next_->forward_propagation(out, worker_index) : out;         //判断是否存在下一层，有则传递[out worker_index]，否则输出out
    }   //const vec_t& forward_propagation(const vec_t& in_raw, size_t worker_index) 



    float_t& weight_at(cnn_size_t in_channel, cnn_size_t out_channel, cnn_size_t kernel_x, cnn_size_t kernel_y) {       //返回权值W中具体位置的值
        return W_[weight_.get_index(kernel_x, kernel_y, in_.depth_ * out_channel + in_channel)];  //index3d<size_T>.get_index() {return (height_ * channel + y) * width_ + x}
    }                                                                                                                                                  // vec_t W_ , 则W_表示具体的值



    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        const vec_t& prev_out = *(prev_out_padded_[index]);                            // const vec_t* prev_out_padded_[];   
        const activation::function& prev_h = prev_->activation_function();      // prev_h 表示上一层的激活函数
        vec_t* prev_delta = (pad_type_ == padding::same) ? &prev_delta_padded_[index] : &prev_delta_[index];        //pad_type为valid，则prev_delta=&prev_delta_[]
        vec_t& dW = dW_[index];         //vec_t dW_[],  vec_t db_[]为具体数值，dW，db为一次TASK处理的vector值
        vec_t& db = db_[index];

        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));          //用0初始化


        // propagate delta to previous layer        #通过卷积层，用delta{l+1}求上一层的delta{l}
        //in_.depth_, [&](int inc)表示 for(inc=0; inc<in_.depth; inc++)
        for_i(in_.depth_, [&](int inc) {                                                    //反向传播
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {                 //out feafure map的数量
                if (!tbl_.is_connected(outc, inc)) continue;

                const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * outc + inc)];        //get_index表示Weight，out_，in_的索引号
                const float_t *pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];                        // pw表示in_.depth*out.depth中的一个weight的首地址，即一个weight
                float_t *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];                   // pdelta_src表示delta_src的指针，共有out_.depth_个
                                                                                                                                                    // prev_delta表示delta_dst即，prev_delta指针，共有

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;                                                       //ppw表示pw指针的具体值，pw表示weight的首地址，即一个weight                                                                          
                        const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];        //ppdelta_src表示delta具体值，pdelta_src表示每次卷积处理的输入patch，
                        float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;     //pad_type=valid, 则pdelta_dst+y*in_padded_.width+x

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[wy * in_padded_.width_ + wx] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });         

        for_i(parallelize_, in_padded_.size(), [&](int i) {
            (*prev_delta)[i] *= prev_h.df(prev_out[i]);                     //df为activation function导数
        });

        // accumulate dw        用delta{l}和x{l-1}对weight求导
        //in_.depth_, [&](int inc)表示 for(inc=0; inc<in_.depth; inc++)
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];       //之前一层的的输出out，即x{l-1}
                        const float_t * delta = &curr_delta[out_.get_index(0, 0, outc)];                    //当前层的误差，即delta{l}

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
                        }
                        dW[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;                  //得到dW
                    }
                }
            }
        });

        // accumulate db
        if (!db.empty()) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const float_t *delta = &curr_delta[out_.get_index(0, 0, outc)];
                db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));             //db就所有该层的delta相加
            }
        }

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(prev_delta_padded_[index], prev_delta_[index]);

        CNN_LOG_VECTOR(curr_delta, "[pc]curr_delta");
        CNN_LOG_VECTOR(prev_delta_[index], "[pc]prev_delta");
        CNN_LOG_VECTOR(dW, "[pc]dW");
        CNN_LOG_VECTOR(db, "[pc]db");

        return prev_->back_propagation(prev_delta_[index], index);
    }       //const vec_t& back_propagation(const vec_t& curr_delta, size_t index)



    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "conv"; }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = weight_.width_ + border_width;                           //每张weight的宽度，用pitch表示，总共有out_.depth_*in_.depth_ 个Weight权重
        const auto width = out_.depth_ * pitch + border_width;                  // 输出宽度
        const auto height = in_.depth_ * pitch + border_width;                   // 输出高度
        const image<>::intensity_t bg_color = 255;          //class image(),    T=uchar, typedef T intensity_t，背景颜色为255

        img.resize(width, height);                                  //用img显示所有的Weights
        img.fill(bg_color);                                               // 用背景色填充img

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());            //minmax为pair<>型

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!tbl_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;                          //每张Weight的左上角坐标
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < weight_.height_; ++y) {                                      
                    for (cnn_size_t x = 0; x < weight_.width_; ++x) {
                        const float_t w = W_[weight_.get_index(x, y, c * in_.depth_ + r)];      //W_[]表示每个Weight中的具体值

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));             //归一化，即根据最大最小值，将值转换为0-255之间
                    }
                }
            }
        }
        return img;
    }       将weight转换为图像



private:
    void init() {
        for (cnn_size_t i = 0; i < CNN_TASK_SIZE; i++) {
            if (pad_type_ == padding::same) {
                prev_out_buf_[i] = new vec_t(in_padded_.size(), float_t(0));
                prev_delta_padded_[i].resize(in_padded_.size(), float_t(0));               
            }
            else {
                prev_out_buf_[i] = nullptr;         //vec_t* prev_out_buf_[]
            }
        }
        if (pad_type_ == padding::same) {
            prev_delta2_padded_.resize(in_padded_.size(), float_t(0));
        }
    }

    cnn_size_t in_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? (in_length + window_size - 1) : in_length;                   //valid，则in_length为原始长度，则in_length
    }

    static cnn_size_t conv_out_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride, padding pad_type) {
        return pad_type == padding::same ? (cnn_size_t)ceil((double)in_length / stride) : (cnn_size_t)ceil((double)(in_length - window_size + 1) / stride);
    }           //padding::same => add zero-padding around image 返回与A同样大小的矩阵        padding::valid => use valid pixels of input   (mA-mB+1)* (mA-mB+1)

    static cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) {
        return conv_out_length(in_width, window_size, w_stride, pad_type) * conv_out_length(in_height, window_size, h_stride, pad_type);
    }           //window_size的情况，计算单个conv_out_dim维度    即conv_out*conv_out

    cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_width, cnn_size_t window_height, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) const {
        return conv_out_length(in_width, window_width, w_stride, pad_type) * conv_out_length(in_height, window_height, h_stride, pad_type);
    }           //window_width, window_height的情况

    void copy_and_unpad_delta(const vec_t& delta, vec_t& dst) {
        if (pad_type_ == padding::valid) {                          //主要考虑valid的情况
            dst = delta;
        }
        else {
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pdst = &dst[in_.get_index(0, 0, c)];
                const float_t *pin = &delta[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pdst += in_.width_, pin += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pdst);
                }
            }
        }
    }

    void copy_and_pad_input(const vec_t& in, int worker_index) {
        vec_t* dst = prev_out_buf_[worker_index];       //  dst = prev_out_buf_[i]，vec_t* prev_out_buf_[];

        if (pad_type_ == padding::valid) {
            prev_out_padded_[worker_index] = &in;           //如果pad_type为valid，则prev_out_padded_[i] = in  使用引用方式
        }
        else {
            // make padded version in order to avoid corner-case in fprop/bprop
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pimg = &(*dst)[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];
                const float_t *pin = &in[in_.get_index(0, 0, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pin += in_.width_, pimg += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pimg);
                }
            }
            prev_out_padded_[worker_index] = prev_out_buf_[worker_index];
        }
    }

    //vec_t* prev_out_padded_[], prev_out_buf_[], prev_delta_padded_[], prev_delta2_padded_
    const vec_t* prev_out_padded_[CNN_TASK_SIZE];                       //上一层输出添加padding
    vec_t* prev_out_buf_[CNN_TASK_SIZE];                                        // 上一层输出缓存
    vec_t  prev_delta_padded_[CNN_TASK_SIZE];                               // 上一层delta添加padded
    vec_t  prev_delta2_padded_;

    //connection_table tbl_
    connection_table tbl_;

    //索引号，在vec_t类型的in, in_padded, out和W_中索引具体的值
    //index3d<cnn_size_t> in_, in_padded_, out_ weight_
    index3d<cnn_size_t> in_;                         
    index3d<cnn_size_t> in_padded_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> weight_;                //weight_实际上只有三个变量，H, W 和 D，起索引作用，(H*C+y)*W+x表示该点的位置

    //enum class padding{valid, same}
    padding pad_type_;

    size_t w_stride_;
    size_t h_stride_;
};

#if 0

#include "tiny_cnn/layers/partial_connected_layer.h"

template<typename Activation = activation::identity>
class convolutional_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * constructing convolutional layer
     *
     * @param in_width     [in] input image width
     * @param in_height    [in] input image height
     * @param window_size  [in] window(kernel) size of convolution
     * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
     * @param out_channels [in] output image channels
     * @param padding      [in] rounding strategy
     *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
     *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
     **/
    convolutional_layer_old(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_size,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        padding pad_type = padding::valid)
    : Base(in_width * in_height * in_channels, out_size(in_width, in_height, window_size, pad_type) * out_channels, 
           sqr(window_size) * in_channels * out_channels, out_channels), 
      in_(in_width, in_height, in_channels), 
      out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
      weight_(window_size, window_size, in_channels*out_channels),
      window_size_(window_size)
    {
        init_connection(connection_table(), pad_type);
    }

    /**
     * constructing convolutional layer
     *
     * @param in_width         [in] input image width
     * @param in_height        [in] input image height
     * @param window_size      [in] window(kernel) size of convolution
     * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
     * @param out_channels     [in] output image channels
     * @param connection_table [in] definition of connections between in-channels and out-channels
     * @param pad_type         [in] rounding strategy 
     *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
     *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
     **/
    convolutional_layer_old(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_size,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        const connection_table& connection_table,
                        padding pad_type = padding::valid)
        : Base(in_width * in_height * in_channels, out_size(in_width, in_height, window_size, pad_type) * out_channels, 
               sqr(window_size) * in_channels * out_channels, out_channels), 
          in_(in_width, in_height, in_channels), 
          out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
          weight_(window_size, window_size, in_channels*out_channels),
          connection_(connection_table),
          window_size_(window_size)
    {
        init_connection(connection_table, pad_type);
        //this->remap();
    }

    image<> output_to_image(size_t worker_index = 0) const {
        return vec2image<unsigned char>(output_[worker_index], out_);
    }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = window_size_ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!connection_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < window_size_; ++y) {
                    for (cnn_size_t x = 0; x < window_size_; ++x) {
                        const float_t w = W_[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "conv"; }

private:
    cnn_size_t out_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? in_length : (in_length - window_size + 1);
    }

    cnn_size_t out_size(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, padding pad_type) const {
        return out_length(in_width, window_size, pad_type) * out_length(in_height, window_size, pad_type);
    }

    void init_connection(const connection_table& table, padding pad_type) {
        cnn_size_t pad = (pad_type == padding::valid) ? 0 : window_size_ / 2;

        for (cnn_size_t inc = 0; inc < in_.depth_; ++inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; ++outc) {
                if (!table.is_connected(outc, inc)) {
                    continue;
                }

                for (cnn_size_t y = 0; y < out_.height_; ++y)
                    for (cnn_size_t x = 0; x < out_.width_; ++x)
                        connect_kernel(inc, outc, x, y, pad);
            }
        }

        for (cnn_size_t outc = 0; outc < out_.depth_; ++outc)
            for (cnn_size_t y = 0; y < out_.height_; ++y)
                for (cnn_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(cnn_size_t inc, cnn_size_t outc, cnn_size_t x, cnn_size_t y, cnn_size_t pad) {

        for (cnn_size_t dy = 0; dy < window_size_; ++dy) {
            if (y + dy < pad) continue;
            if (y + dy - pad >= in_.height_) continue;

            for (cnn_size_t dx = 0; dx < window_size_; ++dx) {
                if (x + dx < pad) continue;
                if (x + dx - pad >= in_.width_) continue;

                this->connect_weight(
                    in_.get_index(x + dx - pad, y + dy - pad, inc), 
                    out_.get_index(x, y, outc), 
                    weight_.get_index(dx, dy, outc * in_.depth_ + inc));
            }
        }
    }

    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> weight_;
    connection_table connection_;
    size_t window_size_;
};

#endif

} // namespace tiny_cnn
