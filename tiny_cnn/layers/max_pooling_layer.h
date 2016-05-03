/*
    Copyright (c) 2015, Taiga Nomi
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
#include "tiny_cnn/layers/partial_connected_layer.h"
#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {
    
template <typename Activation = activation::identity>
class max_pooling_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;
    typedef layer<Activation> Base;

    max_pooling_layer(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t in_channels, cnn_size_t pooling_size)
        : Base(in_width * in_height * in_channels,
        in_width * in_height * in_channels / sqr(pooling_size),
        0, 0),                                // typedef layer<Activation> Base;        layer(in_dim, out_, weight_dim, bias_dim)
        pool_size_(pooling_size),
        stride_(pooling_size),               // 默认情况下，stride_为pooling_size，即不重合
        in_(in_width, in_height, in_channels),
        out_(in_width / pooling_size, in_height / pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size))            //如果相除不为整数，则报错
            pooling_size_mismatch(in_width, in_height, pooling_size);

        init_connection();
    }                                  

    max_pooling_layer(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t in_channels, cnn_size_t pooling_size, cnn_size_t stride)
        : Base(in_width * in_height * in_channels,
        pool_out_dim(in_width, pooling_size, stride) * pool_out_dim(in_height, pooling_size, stride) * in_channels,
        0, 0),                  //当max-pooling有stride的时候，使用pool_out_dim()表示pooling_size
        pool_size_(pooling_size),
        stride_(stride),
        in_(in_width, in_height, in_channels),
        out_(pool_out_dim(in_width, pooling_size, stride), pool_out_dim(in_height, pooling_size, stride), in_channels)
    {
        init_connection();
    }

    size_t fan_in_size() const override {
        return out2in_[0].size();
    }

    size_t fan_out_size() const override {
        return 1;
    }

    size_t connection_size() const override {
        return out2in_[0].size() * out2in_.size();
    }

    virtual const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& out = output_[index];
        vec_t& a = a_[index];
        std::vector<cnn_size_t>& max_idx = out2inmax_[index];      // out对应in_patch的最大标号，与out2in_相同size

        for_(parallelize_, 0, size_t(out_size_), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const auto& in_index = out2in_[i];      //表示out点对应的in_patch，类型为vector<>
                float_t max_value = std::numeric_limits<float_t>::lowest();         //初始化
                
                for (auto j : in_index) {                       //求patch块中的最大值
                    if (in[j] > max_value) {
                        max_value = in[j];
                        max_idx[i] = j;
                    }
                }
                a[i] = max_value;                   //最大值赋给a[i]
            }
        });

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);                                //经过激励函数
        });
        return next_ ? next_->forward_propagation(out, index) : out;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        const vec_t& prev_out = prev_->output(static_cast<int>(index));             // layer_base* prev_;   
        const activation::function& prev_h = prev_->activation_function();          //prev_out上一层的out，prev_h上一层的function
        vec_t& prev_delta = prev_delta_[index];                                     //上一层delta，即所求的delta
        std::vector<cnn_size_t>& max_idx = out2inmax_[index];         // out对应in_patch的最大标号，与out2in_相同size

        for_(parallelize_, 0, size_t(in_size_), [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                cnn_size_t outi = in2out_[i];                               //in2out中，第i个像素点对应outi的值
                prev_delta[i] = (max_idx[outi] == i) ? current_delta[outi] * prev_h.df(prev_out[i]) : float_t(0);   
            }//当max_idx[out]即输出 outi 点对应的 输入in中的第 i 个点，则执行反向传播计算 i 点的delta，其余点delta设置为0
        });
        return prev_->back_propagation(prev_delta_[index], index);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        const vec_t& prev_out = prev_->output(0);
        const activation::function& prev_h = prev_->activation_function();

        for (cnn_size_t i = 0; i < in_size_; i++) {
            cnn_size_t outi = in2out_[i];                   
            prev_delta2_[i] = (out2inmax_[0][outi] == i) ? current_delta2[outi] * sqr(prev_h.df(prev_out[i])) : float_t(0);
        }
        return prev_->back_propagation_2nd(prev_delta2_);
    }

    image<> output_to_image(size_t worker_index = 0) const {
        return vec2image<unsigned char>(output_[worker_index], out_);       //将vec转换为image
    }

    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "max-pool"; }
    size_t pool_size() const {return pool_size_;}

private:
    size_t pool_size_;
    size_t stride_;
    std::vector<std::vector<cnn_size_t> > out2in_; // mapping out => in (1:N)
    std::vector<cnn_size_t> in2out_; // mapping in => out (N:1)
    std::vector<cnn_size_t> out2inmax_[CNN_TASK_SIZE]; // mapping out => max_index(in) (1:1)
    index3d<cnn_size_t> in_;            //in索引
    index3d<cnn_size_t> out_;         //out索引

    static cnn_size_t pool_out_dim(cnn_size_t in_size, cnn_size_t pooling_size, cnn_size_t stride) {
        return (int) std::ceil(((double)in_size - pooling_size) / stride) + 1;
    }

    void connect_kernel(cnn_size_t pooling_size, cnn_size_t outx, cnn_size_t outy, cnn_size_t  c)
    {
        cnn_size_t dxmax = static_cast<cnn_size_t>(std::min((size_t)pooling_size, in_.width_ - outx * stride_));    //stride_=pooling_size，限制kernel的size
        cnn_size_t dymax = static_cast<cnn_size_t>(std::min((size_t)pooling_size, in_.height_ - outy * stride_));

        for (cnn_size_t dy = 0; dy < dymax; dy++) {                 //pooling_size即dxmax, dymax
            for (cnn_size_t dx = 0; dx < dxmax; dx++) {
                cnn_size_t in_index = in_.get_index(static_cast<cnn_size_t>(outx * stride_ + dx),
                                                      static_cast<cnn_size_t>(outy * stride_ + dy), c); //根据一个输出位置(outx, outy)定位in_中的(pooling_size, pooling_size)个点，返回索引号
                cnn_size_t out_index = out_.get_index(outx, outy, c);                   //根据(outx, outy,c)得到输出索引号，(pooling_size, pooling_size)个点对应一个(outx, outy)

                if (in_index >= in2out_.size())
                    throw nn_error("index overflow");
                if (out_index >= out2in_.size())
                    throw nn_error("index overflow");
                in2out_[in_index] = out_index;                  //根据in_index => out_index，匹配关系，即N:1
                out2in_[out_index].push_back(in_index); //vector<std::vector<cnn_size_t> > out2in_;     
            }                                                                     //out2in_[]表示类型为vector的一个点，在这个out点中对应多个in点
        }
    }

    void init_connection()
    {
        in2out_.resize(in_.size());                 // vector<cnn_size_t> in2out_   mapping in => out (N:1)
        out2in_.resize(out_.size());             // vector<cnn_size_t> out2in_   mapping out => in (1:N)
        for (int i = 0; i < CNN_TASK_SIZE; i++)
            out2inmax_[i].resize(out_.size());      //out对应in最大值的索引号

        for (cnn_size_t c = 0; c < in_.depth_; ++c)
            for (cnn_size_t y = 0; y < out_.height_; ++y)
                for (cnn_size_t x = 0; x < out_.width_; ++x)
                    connect_kernel(static_cast<cnn_size_t>(pool_size_),
                                   x, y, c);                            //根据pool_size_, x, y, c初始化connect_kernel
    }

};

} // namespace tiny_cnn
