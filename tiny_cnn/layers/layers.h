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
#include "/deep/tmp/tiny_cnn/tiny_cnn/layers/layer.h"
#include "input_layer.h"

namespace tiny_cnn {

class layers {
public:
    layers() { add(std::make_shared<input_layer>()); }      //构造输出layer, 默认初始化加入input_layer层; ps: shared_ptr<int> p = make_shared<int>(42)

    layers(const layers& rhs) { construct(rhs); }           //构造网络layers

    layers& operator = (const layers& rhs) {            //重载, 拷贝另外一个layers
        layers_.clear();
        construct(rhs);
        return *this;
    }

    void add(std::shared_ptr<layer_base> new_tail) {
        if (tail())  tail()->connect(new_tail);             // 在tail后添加新的tail, 对于<input_layer>, 此tail为空, 则直接push_back
        layers_.push_back(new_tail);                    // 在所有layers中加入new_tail
    }

    bool empty() const { return layers_.size() == 0; }

    layer_base* head() const { return empty() ? 0 : layers_[0].get(); }         //取layers[0], 即第一层layer

    layer_base* tail() const { return empty() ? 0 : layers_[layers_.size() - 1].get(); }    //取layers[n], 即最后一层layer

    template <typename T>
    const T& at(size_t index) const {
        const T* v = dynamic_cast<const T*>(layers_[index + 1].get());        //dynamic_cast是将基类指针或引用转换为派生类的指针或引用
        if (v) return *v;                         //如果v该层存在, 则返回
        throw nn_error("failed to cast");
    }

    const layer_base* operator [] (size_t index) const {            //重载[], 通过layers[]使用
        return layers_[index + 1].get();                            // shared_ptr.get() 表示获得传统的C指针                    
    }

    layer_base* operator [] (size_t index) {            //重载[], 通过layers[]使用
        return layers_[index + 1].get();
    }

    void init_weight() {                                // layers_为vector<shared_ptr<layer_base>>, 则pl为shared_ptr<layer_base>格式
        for (auto pl : layers_)
            pl->init_weight();                          //调用init_weight()初始化
    }

    bool is_exploded() const {
        for (auto pl : layers_)
            if (pl->is_exploded()) return true;     //每层layer是否有解
        return false;
    }

    void divide_hessian(int denominator) {
        for (auto pl : layers_)
            pl->divide_hessian(denominator);        //求解hessian矩阵
    }

    template <typename Optimizer>
    void update_weights(Optimizer *o, size_t worker_size, size_t batch_size) {
        for (auto pl : layers_)                     // pl表示point layer
            pl->update_weight(o, static_cast<cnn_size_t>(worker_size), batch_size);         //逐层update_weight更新权值
    }

    void set_parallelize(bool parallelize) {    // 是否并行
        for (auto pl : layers_)
            pl->set_parallelize(parallelize);       //逐层设置并行化
    }

    // get depth(number of layers) of networks
    size_t depth() const {
        return layers_.size() - 1; // except input-layer        除了输入层, 返回layers的层数
    }

private:
    void construct(const layers& rhs) {                   // 构造网络，逐层加入各层
        add(std::make_shared<input_layer>());       //构造输入层
        for (size_t i = 1; i < rhs.layers_.size(); i++)     //逐层添加网络每层
            add(rhs.layers_[i]);
    }

    std::vector<std::shared_ptr<layer_base>> layers_;           // layers存储各layer, layers元素是用shared_ptr<layer_base>指针表示
                                                                                            // layers为layer的集合，主要对其中各层进行总体操作
};

} // namespace tiny_cnn
