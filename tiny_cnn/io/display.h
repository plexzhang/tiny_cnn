//addapted from boost progress.hpp, made c++11 only//


#ifndef PROGRESS_H
#define PROGRESS_H

#include <iostream>           // for ostream, cout, etc
#include <string>             // for string
#include <chrono>

namespace tiny_cnn {

class timer
{
 public:
    timer():  t1(std::chrono::high_resolution_clock::now()){};          //用当前时间初始化t1;
    double elapsed(){return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();} // 计算经过的片刻数
    void restart(){t1 = std::chrono::high_resolution_clock::now();}
    void start(){t1 = std::chrono::high_resolution_clock::now();}   
    void stop(){t2 = std::chrono::high_resolution_clock::now();}  // t2表示当前时刻的time_point
    double total(){stop();return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();}
    ~timer(){}
 private:
    std::chrono::high_resolution_clock::time_point t1, t2;    //创建两个clock的time_point点t1和t2，t1为起始点，t2为终止点

};


//  progress_display  --------------------------------------------------------//

//  progress_display displays an appropriate indication of 
//  progress at an appropriate place in an appropriate form.

class progress_display
{
 public:
  explicit progress_display( unsigned long expected_count_,
                             std::ostream & os = std::cout,
                             const std::string & s1 = "\n", //leading strings
                             const std::string & s2 = "",
                             const std::string & s3 = "" )
   // os is hint; implementation may ignore, particularly in embedded systems
   :  m_os(os), m_s1(s1), m_s2(s2), m_s3(s3) { restart(expected_count_); }          // _expexcted_count表示data的size即图片数量

  void           restart( unsigned long expected_count_ )
  //  Effects: display appropriate scale
  //  Postconditions: count()==0, expected_count()==expected_count_
  {
    _count = _next_tic_count = _tic = 0;      //重新设置参数
    _expected_count = expected_count_;

    m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
         << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
         << std::endl  // endl implies flush, which ensures display
         << m_s3;
    if ( !_expected_count ) _expected_count = 1;  // prevent divide by zero
  } // restart

  unsigned long  operator+=( unsigned long increment )
  //  Effects: Display appropriate progress tic if needed.
  //  Postconditions: count()== original count() + increment
  //  Returns: count().
  {
    if ( (_count += increment) >= _next_tic_count ) { display_tic(); }
    return _count;
  }

  unsigned long  operator++()           { return operator+=( 1 ); }
  unsigned long  count() const          { return _count; }
  unsigned long  expected_count() const { return _expected_count; }

  private:
  std::ostream &     m_os;  // may not be present in all imps
  const std::string  m_s1;  // string is more general, safer than 
  const std::string  m_s2;  //  const char *, and efficiency or size are
  const std::string  m_s3;  //  not issues

  unsigned long _count, _expected_count, _next_tic_count;
  unsigned int  _tic;
  void display_tic()
  {
    // use of floating point ensures that both large and small counts
    // work correctly.  static_cast<>() is also used several places
    // to suppress spurious compiler warnings. 
    unsigned int tics_needed =                                         // 使用static_cast<T>避免编译错误
      static_cast<unsigned int>(
        (static_cast<double>(_count)/_expected_count)*50.0 );   //计算需要的tics_needed数量
    do { m_os << '*' << std::flush; } while ( ++_tic < tics_needed );   // flush清空缓存区，总共输出tics_needed个*符号
    _next_tic_count = 
      static_cast<unsigned long>((_tic/50.0)*_expected_count);
    if ( _count == _expected_count ) {
      if ( _tic < 51 ) m_os << '*';
      m_os << std::endl;
      }
  } // display_tic

  progress_display &operator = (const progress_display &) = delete;
};

} // namespace

#endif 