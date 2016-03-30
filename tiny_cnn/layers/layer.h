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
#include <sstream>
#include <iomanip>
#include <memory>
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/util/weight_init.h"

#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {


// base class of all kind of NN layers
class layer_base {
public:
    friend void connection_mismatch(const layer_base& from, const layer_base& to);  // friend connection_mismatch可使用layer_base中的成员变量

    virtual ~layer_base() {}

    layer_base(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim)   
        : parallelize_(true), next_(nullptr), prev_(nullptr),
          weight_init_(std::make_shared<weight_init::xavier>()),
          bias_init_(std::make_shared<weight_init::constant>(float_t(0))) {
        set_size(in_dim, out_dim, weight_dim, bias_dim);
    }   //初始化weight与bias

    void connect(std::shared_ptr<layer_base>& tail) {   //tail表示添加的最尾层
        if (out_size() != 0 && tail->in_size() != out_size())   //out_size()为目前层的输出, 判断目前层的输出与新添层输入的匹配
            connection_mismatch(*this, *tail);  //this表示目前层, tail表示新添层
        next_ = tail.get();         //通过链表讲两层相连接
        tail->prev_ = this;
    }   

    void set_parallelize(bool parallelize) {        //设置并行计算
        parallelize_ = parallelize;
    }

    // cannot call from ctor because of pure virtual function call fan_in_size().
    // so should call this function explicitly after ctor           显式使用构造函数
    void init_weight() {
        weight_init_->fill(&W_, static_cast<cnn_size_t>(fan_in_size()),     
                           static_cast<cnn_size_t>(fan_out_size()));
        bias_init_->fill(&b_, static_cast<cnn_size_t>(fan_in_size()),
                         static_cast<cnn_size_t>(fan_out_size()));

        std::fill(Whessian_.begin(), Whessian_.end(), float_t(0));
        std::fill(bhessian_.begin(), bhessian_.end(), float_t(0));
        clear_diff(CNN_TASK_SIZE);
    }

    void divide_hessian(int denominator) {
        for (auto& w : Whessian_) w /= denominator;
        for (auto& b : bhessian_) b /= denominator;
    }       //hessian矩阵操作

    /////////////////////////////////////////////////////////////////////////
    // getter (输出函数)

    const vec_t& output(cnn_size_t worker_index) const { return output_[worker_index]; }        //output索引
    const vec_t& delta(cnn_size_t worker_index) const { return prev_delta_[worker_index]; }     //delta索引
    vec_t& weight() { return W_; }      // typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;
    vec_t& bias() { return b_; }
    vec_t& weight_diff(cnn_size_t index) { return dW_[index]; }         //dW, db为求导
    vec_t& bias_diff(cnn_size_t index) { return db_[index]; }
    bool is_exploded() const { return has_infinite(W_) || has_infinite(b_); }       //判断W和b是否可求
    layer_base* next() { return next_; }
    layer_base* prev() { return prev_; }

    ///< input dimension
    virtual cnn_size_t in_size() const { return in_size_; }

    cnn_size_t in_dim() const { return in_size(); }     //in_dim() = in_size() = in_size_

    ///< output dimension
    virtual cnn_size_t out_size() const { return out_size_; }

    cnn_size_t out_dim() const { return out_size(); }   //out_dim() = out_size() = out_size_

    ///< number of parameters
    virtual size_t param_size() const { return W_.size() + b_.size(); }     //param_size()参数数量

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const = 0;     //每个输入单元的连接数

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const = 0;    //每个输出单元的连接数

    ///< number of connections
    virtual size_t connection_size() const = 0;     //总连接数

    ///< input shape(width x height x depth)
    virtual index3d<cnn_size_t> in_shape() const { return index3d<cnn_size_t>(in_size(), 1, 1); }       //struct index3d(T width, T height, T depth)

    ///< output shape(width x height x depth)
    virtual index3d<cnn_size_t> out_shape() const { return index3d<cnn_size_t>(out_size(), 1, 1); }

    ///< name of layer. should be unique for each concrete class
    virtual std::string layer_type() const = 0;

    virtual activation::function& activation_function() = 0;

    /////////////////////////////////////////////////////////////////////////
    // setter(设置函数)

    template <typename WeightInit>
    layer_base& weight_init(const WeightInit& f) { weight_init_ = std::make_shared<WeightInit>(f); return *this; }      //通过make_shared()函数分配和使用动态内存, 
                                                                                                                                                                                        //返回指向此对象的shared_ptr, 最终返回layer_base

    template <typename BiasInit>
    layer_base& bias_init(const BiasInit& f) { bias_init_ = std::make_shared<BiasInit>(f); return *this; }

    template <typename WeightInit>
    layer_base& weight_init(std::shared_ptr<WeightInit> f) { weight_init_ = f; return *this; }  //重载, 参数为shared_ptr

    template <typename BiasInit>
    layer_base& bias_init(std::shared_ptr<BiasInit> f) { bias_init_ = f; return *this; }

    /////////////////////////////////////////////////////////////////////////
    // save/load
    virtual void save(std::ostream& os) const {
        if (is_exploded()) throw nn_error("failed to save weights because of infinite weight");
        for (auto w : W_) os << w << " ";
        for (auto b : b_) os << b << " ";
    }

    virtual void load(std::istream& is) {
        for (auto& w : W_) is >> w;
        for (auto& b : b_) is >> b;
    }

    /////////////////////////////////////////////////////////////////////////
    // visualize

    ///< visualize latest output of this layer
    ///< default implementation interpret output as 1d-vector,
    ///< so "visual" layer(like convolutional layer) should override this for better visualization.
    virtual image<> output_to_image(size_t worker_index = 0) const {
        return vec2image<unsigned char>(output_[worker_index]);
    }       //可视化, 默认情况下, 输出为1维向量,  类似卷积层的可视化, 通过override重写

    /////////////////////////////////////////////////////////////////////////
    // fprop/bprop(前/后向传播)

    /**
     * return output vector
     * output vector must be stored to output_[worker_index]
     **/
    virtual const vec_t& forward_propagation(const vec_t& in, size_t worker_index) = 0;     //根据输入, 返回输出, 存储为output_[]

    /**
     * return delta of previous layer (delta=\frac{dE}{da}, a=wx in fully-connected layer)
     * delta must be stored to prev_delta_[worker_index]
     **/
    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t worker_index) = 0; //  根据当前delta, 返回前一delta

    /**
     * return delta2 of previous layer (delta2=\frac{d^2E}{da^2}, diagonal of hessian matrix)
     * it is never called if optimizer is hessian-free
     **/
    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) = 0;     //hessian矩阵对角线

    // called afrer updating weight
    virtual void post_update() {}      

    /**
     * notify changing context (train <=> test)
     **/
     virtual void set_context(net_phase ctx) { CNN_UNREFERENCED_PARAMETER(ctx); }       //设置train 或 test

    template <typename Optimizer>
    void update_weight(Optimizer *o, cnn_size_t worker_size, cnn_size_t batch_size) {       //更新, 其中worker_size表示线程总数
        if (W_.empty()) return;

        merge(worker_size, batch_size);

        CNN_LOG_VECTOR(W_, "[W-before]");
        CNN_LOG_VECTOR(b_, "[db-before]");

        o->update(dW_[0], Whessian_, W_);
        o->update(db_[0], bhessian_, b_);

        CNN_LOG_VECTOR(W_, "[W-updated]");
        CNN_LOG_VECTOR(b_, "[db-updated]");

        clear_diff(worker_size);
        post_update();
    }

    bool has_same_weights(const layer_base& rhs, float_t eps) const {
        if (W_.size() != rhs.W_.size() || b_.size() != rhs.b_.size())
            return false;

        for (size_t i = 0; i < W_.size(); i++)
          if (std::abs(W_[i] - rhs.W_[i]) > eps) return false;
        for (size_t i = 0; i < b_.size(); i++)
          if (std::abs(b_[i] - rhs.b_[i]) > eps) return false;

        return true;
    }

protected:
    cnn_size_t in_size_;        //输入size
    cnn_size_t out_size_;       //输出size
    bool parallelize_;

    layer_base* next_;
    layer_base* prev_;
    vec_t a_[CNN_TASK_SIZE];          // w * x          #define CNN_TASK_SIZE 100 or 8
    vec_t output_[CNN_TASK_SIZE];     // last output of current layer, set by fprop
    vec_t prev_delta_[CNN_TASK_SIZE]; // last delta of previous layer, set by bprop
    vec_t W_;          // weight vector         W_和dW_不同, 只表示一层的Weight
    vec_t b_;          // bias vector

    /** contribution to derivative of loss function with respect to weights of this layer,
        indexed by worker / thread */
    vec_t dW_[CNN_TASK_SIZE];       //  表示有CNN_TASK_SIZE个vecotr<float_t>, 有work thread索引

    /** contribution to derivative of loss function with respect to bias terms of this layer,
        indexed by worker / thread */
    vec_t db_[CNN_TASK_SIZE];

    vec_t Whessian_; // diagonal terms of hessian matrix
    vec_t bhessian_;
    vec_t prev_delta2_; // d^2E/da^2
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    // vec_t    a_[],    output_[],    prev_delta_[],     dW_[],          db_[]
    // vec_t    W_,     b_,               Whessian,          bheassian,   prev_delta2

private:
    /** sums contributions to gradient (of the loss function with respect to weights and
        bias) as calculated by individual threads */
    void merge(cnn_size_t worker_size, cnn_size_t batch_size) {                 //worker_size表示线程数量
        for (cnn_size_t i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&dW_[i][0],
                static_cast<cnn_size_t>(dW_[i].size()), &dW_[0][0]);
        for (cnn_size_t i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&db_[i][0],
                static_cast<cnn_size_t>(db_[i].size()), &db_[0][0]);

        std::transform(dW_[0].begin(), dW_[0].end(), dW_[0].begin(), [&](float_t x) { return x / batch_size; });
        std::transform(db_[0].begin(), db_[0].end(), db_[0].begin(), [&](float_t x) { return x / batch_size; });

        CNN_LOG_VECTOR(dW_[0], "[dW-merged]");
        CNN_LOG_VECTOR(db_[0], "[db-merged]");
    }

    void clear_diff(size_t worker_size) {
        for (size_t i = 0; i < worker_size; i++) {
            std::fill(dW_[i].begin(), dW_[i].end(), float_t(0));
            std::fill(db_[i].begin(), db_[i].end(), float_t(0));
        }
    }

    void set_size(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim) {
        in_size_ = in_dim;
        out_size_ = out_dim;

        W_.resize(weight_dim);                  //调整dim, 此五句作用与下五句作用相同
        b_.resize(bias_dim);
        Whessian_.resize(weight_dim);
        bhessian_.resize(bias_dim);
        prev_delta2_.resize(in_dim);

        for (auto& o : output_)     o.resize(out_dim);  //output_为 vector<float_t> output_[CNN_TASK_SIZE], 其中output_[0]为vector, 则for(output[0]~output[CNN_TASK_SIZE])
        for (auto& a : a_)          a.resize(out_dim);
        for (auto& p : prev_delta_) p.resize(in_dim);
        for (auto& dw : dW_) dw.resize(weight_dim);
        for (auto& db : db_) db.resize(bias_dim);
    }
};

template<typename Activation>
class layer : public layer_base {
public:
    layer(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim)        
        : layer_base(in_dim, out_dim, weight_dim, bias_dim) {}

    activation::function& activation_function() override { return h_; }
protected:
    Activation h_;      //typedef activation::identity Activation
};

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const layer_base& v) {
    v.save(os);
    return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, layer_base& v) {
    v.load(os);
    return os;
}

// error message functions

inline void connection_mismatch(const layer_base& from, const layer_base& to) {
    std::ostringstream os;

    os << std::endl;
    os << "output size of Nth layer must be equal to input of (N+1)th layer" << std::endl;
    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:" << from.in_size() << "(" << from.in_shape() << "), " << 
                                                "out:" << from.out_size() << "(" << from.out_shape() << ")" << std::endl;
    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:" << to.in_size() << "(" << to.in_shape() << "), " <<
                                             "out:" << to.out_size() << "(" << to.out_shape() << ")" << std::endl;
    os << from.out_size() << " != " << to.in_size() << std::endl;
    std::string detail_info = os.str();

    throw nn_error("layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const layer_base& layer, const vec_t& data) {
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << std::endl;
    os << "network dimension: " << layer.in_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;

    std::string detail_info = os.str();

    throw nn_error("input dimension mismath!" + detail_info);
}

inline void pooling_size_mismatch(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t pooling_size) {
    std::ostringstream os;

    os << std::endl;
    os << "WxH:" << in_width << "x" << in_height << std::endl;
    os << "pooling-size:" << pooling_size << std::endl;

    std::string detail_info = os.str();

    throw nn_error("width/height must be multiples of pooling size" + detail_info);
}

} // namespace tiny_cnn
