
/** @file numeric.cpp
 *
 *  This file contains the interface to the underlying bignum package.
 *  Its most important design principle is to completely hide the inner
 *  working of that other package from the user of GiNaC.  It must either 
 *  provide implementation of arithmetic operators and numerical evaluation
 *  of special functions or implement the interface to the bignum package. */

/*
 *  This is a modified version of code included with Ginac.  
 *  The modifications and modified version is:
 * 
 *      GiNaC-SAGE Copyright (C) 2008 William Stein
 *      Copyright (C) 2009 Burcin Erocal <burcin@erocal.org>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */


/*  The original copyright:
 * 
 *  GiNaC Copyright (C) 1999-2008 Johannes Gutenberg University Mainz, Germany
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
#include "numeric.h"
#include "constant.h"
#include "ex.h"
#include "operators.h"
#include "archive.h"
#include "tostring.h"
#include "utils.h"

#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <limits>

#include <cmath>

// Call the Python function f on *this as input and return the result
// as a PyObject*.
#define PY_RETURN(f)  PyObject *a = Number_T_to_pyobject(*this);		 \
  PyObject *ans = f(a);						 \
  Py_DECREF(a);                                                  \
  if (!ans) py_error("error calling function");			 \
  return ans; 

#define PY_RETURNO(f) PyObject *ans = f(this->pyobj);						 \
  if (!ans) py_error("error calling function");			 \
  return new PyNumber(ans); 

// Call the Python function f on *this and return the result
// as a PyObject*.
#define PY_RETURN2(f, b)  PyObject *aa = Number_T_to_pyobject(*this);	 \
  PyObject* bb = Number_T_to_pyobject(b);				 \
  PyObject *ans = f(aa, bb);					 \
  if (!ans) py_error("error calling function");			 \
  Py_DECREF(aa); Py_DECREF(bb); return ans; 

#define PY_RETURNO2(f, b) PyObject *ans = f(b,this->pyobj);					 \
  if (!ans) py_error("error calling function");			 \
  return new PyNumber(ans); 

#define PY_RETURNO2a(f, b) PyObject *ans = f(pyobj,b);					 \
  if (!ans) py_error("error calling function");			 \
  return new PyNumber(ans); 



// Call the Python functin f on *this and b, and get back
// a 2-tuple (z,w).  Set c = w, where c should be a 
// reference in the caller, and return z.  This is used
// to return two inputs from a Python function call.  See
// its usage in code below. 
#define PY_RETURN3(f, b, c)						\
  PyObject *aa = Number_T_to_pyobject(*this);					\
  PyObject* bb = Number_T_to_pyobject(b);					\
  PyObject *ans = f(aa, bb);						\
  if (!ans) py_error("error calling function");				\
  if (!PyTuple_CheckExact(ans) || PyTuple_GET_SIZE(ans) != 2) 		\
    py_error("error calling function -- return not a 2-tuple.");	\
  PyObject* z =  PyTuple_GET_ITEM(ans, 0); Py_INCREF(z);                \
  PyObject* w =  PyTuple_GET_ITEM(ans, 1); Py_INCREF(w);          \
  c = w;                                                         \
  Py_DECREF(aa); Py_DECREF(bb); Py_DECREF(ans);                   \
  return z;

#define PY_RETURNO3(f, b, c)	PyObject *ans = f(b, this->pyobj);						\
  if (!ans) py_error("error calling function");				\
  if (!PyTuple_CheckExact(ans) || PyTuple_GET_SIZE(ans) != 2) 		\
    py_error("error calling function -- return not a 2-tuple.");	\
  PyObject* z =  PyTuple_GET_ITEM(ans, 0);  Py_INCREF(z); \
  PyObject* w =  PyTuple_GET_ITEM(ans, 1);  Py_INCREF(w);\
  c=w; \
  return new PyNumber(z); 


//#define DEBUG
//#define VERBOSE

#ifdef DEBUG
#define todo(s) std::cerr << "TODO: " << s << std::endl;
#define stub(s) { std::cerr << "Hit STUB: " << s << std::endl; throw std::runtime_error("stub"); }
#define fake(s) std::cerr << "fake: " << s << std::endl;
#define ASSERT(s, msg) if (!s) { std::cerr << "Failed assertion: " << msg << std::endl; }
#else
#define todo(s)
#define stub(s) { std::cerr << "** Hit STUB**: " << s << std::endl; throw std::runtime_error("stub"); }
#define fake(s)
#endif

#ifdef VERBOSE
#define verbose(s) std::cerr << s << std::endl;
#define verbose2(s,t) std::cerr << s << " " << t << std::endl;
#define verbose3(s,t,u) std::cerr << s << " " << t << ", " << u << std::endl;
#else
#define verbose(s)
#define verbose2(s,t)
#define verbose3(s,t,u)
#endif


//////////////////////////////////////////////////////////////
// Python Interface
//////////////////////////////////////////////////////////////
void py_error(const char* s) {
  if (PyErr_Occurred()) {
    throw std::runtime_error("");
  }
}

// The following variable gets changed to true once
// this library has been imported by the Python
// interpreter.  This is because the interpreter calls
// ginac_pyinit_I below, which sets this to true.
// Once this is done we can call all the py_* functions
// defined in Cython modules, which is often much faster
// than what we can do without those. 
static bool initialized = false;

static PyObject* pyfunc_Float = 0;
void ginac_pyinit_Float(PyObject* f) {
  Py_INCREF(f);
  pyfunc_Float = f;
}

void ginac_pyinit_I(PyObject* z) {
  initialized = true;
  Py_INCREF(z);
  GiNaC::I = z;  // I is a global constant defined below.
}

static PyObject* pyfunc_Integer = 0;
void ginac_pyinit_Integer(PyObject* f) {
  Py_INCREF(f);
  pyfunc_Integer = f;
}

PyObject* Integer(const long int& x) {
  if (initialized) 
    return GiNaC::py_funcs.py_integer_from_long(x);
  
  // Slow version since we can't call Cython-exported code yet.
  PyObject* m = PyImport_ImportModule("sage.rings.integer");
  if (!m)
    py_error("Error importing sage.rings.integer");
  PyObject* Integer = PyObject_GetAttrString(m, "Integer");
  if (!Integer)
    py_error("Error getting Integer attribute");
  PyObject* ans = PyObject_CallFunction(Integer, "l", x);
  //incref maybe?
  Py_DECREF(m);
  Py_DECREF(Integer);
  return ans;
}  

PyObject* Rational(const long int& n, const long int& d) {
  return PyNumber_Divide(Integer(n), Integer(d));
}  

namespace GiNaC {

PyObject* ZERO = PyInt_FromLong(0);   // todo: never freed
PyObject* ONE  = PyInt_FromLong(1);   // todo: never freed
PyObject* TWO  = PyInt_FromLong(2);   // todo: never freed

 
  numeric I; 
///////////////////////////////////////////////////////////////////////////////
// class Number
///////////////////////////////////////////////////////////////////////////////

Number* PyNumber::vadd(Number* n)
{ return n->add(this); 
}
Number* PyNumber::add(PyNumber* n)
{ return new PyNumber(PyNumber_Add(n->pyobj,this->pyobj)); 
}

Number* PyNumber::vsub(Number* n)
{ return n->sub(this); 
}
Number* PyNumber::sub(PyNumber* n)
{ return new PyNumber(PyNumber_Subtract(n->pyobj,this->pyobj)); 
}

Number* PyNumber::vmul(Number* n)
{  
  return n->mul(this); 
}
Number* PyNumber::mul(PyNumber* n)
{  
  PyObject* res= PyNumber_Multiply(n->pyobj,this->pyobj); 
  Py_INCREF(res);
  Number* ret=new PyNumber(res);
  return ret;
}

Number* PyNumber::vdiv(Number* n)
{  
  return n->div(this); 
}
void helo(char c)
{
}

Number* PyNumber::div(PyNumber* n)
{  
  if (PyObject_Compare(pyobj, ONE) == 0) {
//      helo('a');
	    return n;
	}

	if (PyInt_Check(n->pyobj))  
  {
	  if(PyInt_Check(this->pyobj))
    {
      // This branch happens at startup.
      PyObject* io = Rational(PyInt_AS_LONG(n->pyobj),  
                 PyInt_AS_LONG(this->pyobj));
      // I don't 100% understand why I have to incref this, 
      // but if I don't, Sage crashes on exit.
      Py_INCREF(io);
      return new PyNumber(io); 
	  } 
    else if (PyLong_Check(this->pyobj))
    {
      PyObject* d = py_funcs.py_integer_from_python_obj(this->pyobj);
      PyObject* ans = PyNumber_Divide(n->pyobj,d);
      Py_DECREF(d);
      return new PyNumber(ans);
	  }
	} 
  else if (PyLong_Check(n->pyobj)) 
  {
      PyObject* nn = py_funcs.py_integer_from_python_obj(n->pyobj);
	    PyObject* ans = PyNumber_Divide(this->pyobj,nn);
	    Py_DECREF(nn);
	    return new PyNumber(ans);
	}
  return new PyNumber(PyNumber_Divide(n->pyobj,this->pyobj));
}

Number* PyNumber::vpow(Number* n)
{ //  
  return n->pow(this);
}
Number* PyNumber::pow(PyNumber* n)
{  
  if PyInt_Check(this->pyobj) 
  {  
    PyObject* io = Integer(PyInt_AsLong(this->pyobj));
    PyObject* r = PyNumber_Power(n->pyobj, io, Py_None);
    Py_DECREF(io);
    return new PyNumber(r);
  }
  return new PyNumber(PyNumber_Power(n->pyobj,pyobj, Py_None));
}

///////////////////////////////////////////////////////////////////////////////
// class PyNumber
///////////////////////////////////////////////////////////////////////////////
PyNumber::PyNumber()
{ verbose("PyNumber::PyNumber()");
  if(!(pyobj=PyInt_FromLong(0)))
  { py_error("Error creating 0 number");
  }
}

PyNumber::PyNumber(const int& x)
{  
  verbose("PyNumber::PyNumber(const int& x)");
  if (!(pyobj = PyInt_FromLong(x)))
  { py_error("Error creating int");
  }
}

PyNumber::PyNumber(const long int& x) 
{ verbose("Number_T::Number_T(const long int& x)");
  if (!(pyobj = PyInt_FromLong(x)))
  { py_error("Error creating long int");
  }
}

PyNumber::PyNumber(const unsigned int& x) 
{ 
  // TODO !!!! -- these won't work since Integer assumes
  // input is signed!!!
  verbose("PyNumber::PyNumber(const unsigned int& x)");
  pyobj = Integer(x);
  Py_INCREF(pyobj);
}

PyNumber::PyNumber(const unsigned long& x) 
{ 
  verbose("PyNumber::PyNumber(const unsigned long& x)");
  pyobj = Integer(x);
  Py_INCREF(pyobj);
}

PyNumber::PyNumber(const double& x) 
{ 
  verbose("PyNumber::PyNumber(const double& x)");
  if (!(pyobj =  PyFloat_FromDouble(x)))
  { py_error("Error creating double");
  }
}

PyNumber::PyNumber(const char *s) 
{ 
  verbose("PyNumber::PyNumber(const char* x)");
  double d;
  sscanf(s,"%lf",&d);
  if (!(pyobj =  PyFloat_FromDouble(d)))
  {  py_error("Error creating double");
  }
}


PyNumber::PyNumber(Number* n) 
{  
  verbose("Number_T::Number_T(const Number_T& x)");
  PyNumber *p=dynamic_cast<PyNumber*>(n);
  if(p!=NULL)
  {  
    pyobj=p->pyobj;
    Py_INCREF(pyobj);
  }
  else
  { stub("PyNumber(Number* n) needs support for other types");
  }   

}
 
PyNumber::PyNumber(PyObject* o) 
{  
  verbose("Number_T::Number_T(PyObject* o)");
  if(!o) 
  { py_error("Error");
  }
  else
  {  
    this->pyobj=o;
  }
//  
}
/*
PyNumber::PyNumber(const char* s) {
  double d;
  // We should never use this. 
  verbose("Number_T(const char* s)");
  t = DOUBLE;
  sscanf(s, "%lf", &d);
  PyNumber(d);
}*/

PyNumber::PyNumber(const archive_node &n, lst &sym_lst)
{ 
  unsigned int t_tmp;
  if(!n.find_unsigned(std::string("T"), t_tmp))
  { 
    throw std::runtime_error("archive error: cannot read type info");
  }
  Type t=Type(t_tmp);
  std::string str;
  PyObject *arg;
  switch(t)
  { 
    case PYOBJECT:
      if(!n.find_string("S",str))
      { 
        throw(std::runtime_error("archive error: cannot read pyobject data"));
      }
      arg = Py_BuildValue("s#",str.c_str(), str.size());
      pyobj=py_funcs.py_loads(arg);
      Py_DECREF(arg);
      if (PyErr_Occurred())
      {
        throw(std::runtime_error("archive error: caught exception in py_loads"));
      }
      Py_INCREF(pyobj);
      return;
    default:
      std::cout<<"PyNumber::PyNumber missing case";
      return;
  }
}
PyNumber::~PyNumber()
{ Py_DECREF(this->pyobj);
}

PyObject* PyNumber::pyobject()
{
  Py_INCREF(pyobj);
  return pyobj;
}

//operator <<
std::string PyNumber::str()
{ PyObject* o = PyObject_Repr(pyobj);
  std::string ret;
  if (!o) 
  { PyErr_Clear();
    throw(std::runtime_error("PyNumber::print"));
  } 
  else 
  { ret=PyString_AsString(o);
	  Py_DECREF(o);
  }
  return ret;
}
void PyNumber::archive(archive_node &n) const {
	// store type information
	n.add_unsigned("T", PYOBJECT); 

	// create a string representation of this object
	std::string *tstr;
  tstr = py_funcs.py_dumps(pyobj);
  if (PyErr_Occurred()) {
      throw(std::runtime_error("archive error: exception in py_dumps"));
  }
	n.add_string("S",*tstr);
	delete tstr;
}
// use copy constructor instead of operator=
double PyNumber::to_double() const 
{ verbose("to_double");
  double d = PyFloat_AsDouble(pyobj);
  if (d == -1 && PyErr_Occurred())
  { py_error("Error converting to a double.");
  }
  return d;
}

int PyNumber::to_int() const 
{ verbose("to_int");
  long n = PyInt_AsLong(pyobj);
  if (((int)n != n) || (n == -1 && PyErr_Occurred()))
  { py_error("Error converting to a long.");
  }
  return n;
}

long int PyNumber::to_long_int() const 
{ 
  verbose("to_long_int");
  long int n = PyInt_AsLong(pyobj);
  if (n == -1 && PyErr_Occurred()) 
  { PyErr_Print();
    py_error("Overfloat converting to long int");
  }
  return n;
}

unsigned PyNumber::hash() const 
{ 
  verbose("hash");
  long res = PyObject_Hash(pyobj);
  if (res == -1 && PyErr_Occurred()) 
  {
    throw(std::runtime_error("Number_T::hash() python function (__hash__) raised exception"));
  }
  return res;
}
bool PyNumber::vequal(Number* n) 
{ 
  return n->equal(this); 
}
bool PyNumber::equal(PyNumber *n) const 
{  
  return py_funcs.py_is_equal(n->pyobj, pyobj);
}

bool PyNumber::vnot_equal(Number* n) 
{ 
  return n->not_equal(this); 
}

bool PyNumber::not_equal(PyNumber* n) const 
{ 
  return (!py_funcs.py_is_equal(n->pyobj, pyobj));
}

int PyNumber::vcompare_same_type(Number *n) 
{ 
  return n->compare_same_type(this);
}

int PyNumber::compare_same_type(PyNumber *n) const 
{ verbose("compare_same_type");
  int result;
  if (PyObject_Cmp(n->pyobj, pyobj, &result) == -1) 
  { py_error("compare_same_type");
  }
  return result;
}

bool PyNumber::vless_equal(Number* n)
{ 
  return n->less_equal(this);
}

bool PyNumber::less_equal(PyNumber *n) const 
{ int result;
  if (PyObject_Cmp(n->pyobj, pyobj, &result) == -1) 
  { py_error("less_or_equal");
  }
  return (result <= 0);
}

bool PyNumber::vgreater_equal(Number* n)
{ 
  return n->greater_equal(this);
}
bool PyNumber::greater_equal(PyNumber* n) const 
{ 
  verbose("greater_or_equal");
  int result;
  if (PyObject_Cmp(n->pyobj, pyobj, &result) == -1) 
  { py_error(">=");
  }
  return (result >= 0);
}

bool PyNumber::vless(Number* n)
{ 
  return n->less(this);
}

bool PyNumber::less(PyNumber *n)const 
{ verbose("less_than");
  int result;
  if (PyObject_Cmp(n->pyobj, pyobj, &result) == -1) 
  { py_error("less_than");
  }
  return (result < 0);
}

bool PyNumber::vgreater(Number* n)
{ 
  return n->greater(this);
}

bool PyNumber::greater(PyNumber* n)const  
{ 
  int result;
  if (PyObject_Cmp(n->pyobj, pyobj, &result) == -1) 
  { py_error("greater_than");
  }
  return (result > 0);
}

int PyNumber::csgn() const 
{ verbose("csgn");
  int result;
  if (is_real()) 
  { if (PyObject_Cmp(pyobj, ZERO, &result) == -1)
    { py_error("csgn");
    }
  } 
  else 
  { PyObject *t = py_funcs.py_real(pyobj);
    if (PyObject_Cmp(t, ZERO, &result) == -1)
      py_error("csgn");
    if (result == 0) 
    { Py_DECREF(t);
      t = py_funcs.py_imag(pyobj);
      if (PyObject_Cmp(t, ZERO, &result) == -1)
        py_error("csgn");
      Py_DECREF(t);
    }
  }
  return result;
}


bool PyNumber::is_zero()  const { 
  verbose("is_zero");
  int a;
    a = PyObject_Not(pyobj);
    if (a==-1)
py_error("is_zero");
    return a;
}

bool PyNumber::is_positive() const  { 
  verbose("is_positive");
  bool n;
    n = is_real() && (PyObject_Compare(pyobj, ZERO) > 0);
    if (PyErr_Occurred()) 
py_error("is_positive");
    return n;
}

bool PyNumber::is_negative()  const { 
  verbose("is_negative");
  bool n;
    n = is_real() && (PyObject_Compare(pyobj, ZERO) < 0);
    if (PyErr_Occurred()) 
py_error("is_negative");
    return n;
}

bool PyNumber::is_integer() const 
{ return py_funcs.py_is_integer(pyobj);
}

bool PyNumber::is_cinteger()  const 
{ verbose("is_crational");
  return py_funcs.py_is_cinteger(pyobj);
}

bool PyNumber::is_pos_integer()  const 
{ verbose("is_pos_integer");
  return (this->is_integer() && this->is_positive());
}

bool PyNumber::is_nonneg_integer()  const 
{ verbose("is_nonneg_integer");
  bool n = (is_integer() && (PyObject_Compare(pyobj, ZERO) >= 0));
  if (PyErr_Occurred()) 
py_error("is_nonneg_integer");
    return n;
}

bool PyNumber::is_even()  const 
{ verbose("is_even");
  if (!is_integer()) 
    return false;
  return py_funcs.py_is_even(pyobj);
}

bool PyNumber::is_odd()  const 
{ verbose("is_odd");  
  return !is_even();
}

bool PyNumber::is_prime()  const 
{ verbose("is_prime");
  return py_funcs.py_is_prime(pyobj);
}
 
bool PyNumber::is_rational()  const 
{ verbose("is_rational");
  return py_funcs.py_is_rational(pyobj);
}

bool PyNumber::is_crational()  const 
{  
  verbose("is_crational");
  return py_funcs.py_is_crational(pyobj);
}

bool PyNumber::is_real()  const 
{ verbose("is_real");
  return py_funcs.py_is_real(pyobj);
}

int PyNumber::get_parent_char() const 
{ verbose("get_parent_char");
  int c = py_funcs.py_get_parent_char(pyobj);
  if (c == -1) 
  { py_error("error in py_get_parent_char");
  }
  return c;
}

Number* PyNumber::numer() 
{  
  verbose("numer");
  PyObject* a = py_funcs.py_numer(pyobj);
  if (!a)
  { py_error("numer");
  }
  return new PyNumber(a);
}

Number* PyNumber::denom() 
{  
  verbose("denom");
  PyObject* a = py_funcs.py_denom(pyobj);
  if (!a) 
  { py_error("denom");
  }
  return new PyNumber(a);
}
Number* PyNumber::real() 
{
  verbose("real");
  PyObject* a = py_funcs.py_real(pyobj);
  if (!a) 
  { py_error("real");
  }
  return new PyNumber(a);
}

Number* PyNumber::imag() 
{ 
  verbose("imag");
  //
  PyObject* a = py_funcs.py_imag(pyobj);
  if (!a) 
  { 
    
     py_error("imag");
  }
  return new PyNumber(a);
}

Number* PyNumber::conjugate() 
{ PyObject* a= py_funcs.py_conjugate(pyobj);
  if(!a)
  { py_error("conjugate");
  }
  return new PyNumber(a);
}

Number* PyNumber::step()  
{ PY_RETURNO(py_funcs.py_step);
}

Number* PyNumber::fibonacci()  {
  PY_RETURNO(py_funcs.py_fibonacci);
}
Number* PyNumber::evalf(PyObject *parent)  {
  PY_RETURNO2a(py_funcs.py_float,parent);
}


Number* PyNumber::sin()  {
  PY_RETURNO(py_funcs.py_sin);
}

Number* PyNumber::cos()  {
  PY_RETURNO(py_funcs.py_cos);
}

Number* PyNumber::zeta()  {
  PY_RETURNO(py_funcs.py_zeta);
}

Number* PyNumber::exp()  {
  PY_RETURNO(py_funcs.py_exp);
}

Number* PyNumber::log()  {
  PY_RETURNO(py_funcs.py_log);
}

Number* PyNumber::tan()  {
  PY_RETURNO(py_funcs.py_tan);
}

Number* PyNumber::asin()  {
  
  PY_RETURNO(py_funcs.py_asin);
}
  
Number* PyNumber::acos()  {
  PY_RETURNO(py_funcs.py_acos);
}

Number* PyNumber::atan()  
{ if(is_real() && real()->is_zero() && std::abs(imag()->equal(new PyNumber(ONE))))
    throw pole_error("atan(): logarithmic pole",0);
  PY_RETURNO(py_funcs.py_atan);
}

Number* PyNumber::vatan(const Number *n)  {
  Number* m=const_cast<Number*>(n);
  return m->atan(this);
}

Number* PyNumber::atan(PyNumber *n)  {
  PY_RETURNO2(py_funcs.py_atan2, n->pyobj);
}

Number* PyNumber::sinh()  {
  PY_RETURNO(py_funcs.py_sinh);
}

Number* PyNumber::cosh()  {
  PY_RETURNO(py_funcs.py_cosh);
}

Number* PyNumber::tanh()  {
  PY_RETURNO(py_funcs.py_tanh);
}

Number* PyNumber::asinh()  {
  PY_RETURNO(py_funcs.py_asinh);
}

Number* PyNumber::acosh()  {
  PY_RETURNO(py_funcs.py_acosh);
}

Number* PyNumber::atanh()  {
  PY_RETURNO(py_funcs.py_atanh);
}

Number* PyNumber::vLi(const Number *n, PyObject* parent)
{ 
  Number* m=const_cast<Number*>(n);
  m->Li(this,parent); 
}

Number* PyNumber::Li(PyNumber *n, PyObject* parent)  {
  PyObject *ans = py_funcs.py_li(pyobj,n->pyobj, parent);             
  if (!ans) py_error("error calling function");
  return new PyNumber(ans); 
}

Number* PyNumber::Li2()  {
  PY_RETURNO(py_funcs.py_li2);
}

Number* PyNumber::lgamma()  {
  PY_RETURNO(py_funcs.py_lgamma);
}

Number* PyNumber::tgamma()  {
  PY_RETURNO(py_funcs.py_tgamma);
}

Number* PyNumber::psi()  {
  PY_RETURNO(py_funcs.py_psi);
}

Number* PyNumber::vpsi(Number* n)  {
  return n->psi(this);
}

Number* PyNumber::psi(PyNumber* n)  {
  PY_RETURNO2(py_funcs.py_psi2, n->pyobj);
}

Number* PyNumber::factorial()  {
  PY_RETURNO(py_funcs.py_factorial);
}

Number* PyNumber::doublefactorial()  {
  PY_RETURNO(py_funcs.py_doublefactorial);
}

Number* PyNumber::isqrt()  {
  PY_RETURNO(py_funcs.py_isqrt);
}

Number* PyNumber::sqrt()  {
  PY_RETURNO(py_funcs.py_sqrt);
}

Number* PyNumber::abs()  {
  PY_RETURNO(py_funcs.py_abs);
}

Number* PyNumber::vmod(Number *n)  {
  return n->mod(this); 
}
Number* PyNumber::mod(PyNumber *n)  {
  PY_RETURNO2(py_funcs.py_mod, n->pyobj);
}


Number* PyNumber::vsmod(Number *n)  {
  return n->smod(this);
}

Number* PyNumber::smod(PyNumber *n)  {
  PY_RETURNO2(py_funcs.py_smod, n->pyobj);
}

Number* PyNumber::virem(Number *n)  {
  return n->irem(this);
}

Number* PyNumber::irem(PyNumber *n)  {
  PY_RETURNO2(py_funcs.py_irem, n->pyobj);
}

Number* PyNumber::viquo(Number *n)  {
  return n->iquo(this);
}

Number* PyNumber::iquo(PyNumber *n)  
{
  PY_RETURNO2(py_funcs.py_iquo, n->pyobj);
}

Number* PyNumber::viquo(Number *b,Number *r)  
{
  b->iquo(this,r);
}

Number* PyNumber::iquo(PyNumber *b,Number *r)  
{
  PyNumber* r_py=dynamic_cast<PyNumber*>(r);
  if(!r_py)
  { throw std::runtime_error("iquo");
  }
  PY_RETURNO3(py_funcs.py_iquo2, b->pyobj, r_py->pyobj);
}

int PyNumber::int_length() const {
  int n = py_funcs.py_int_length(pyobj);
  if (n == -1)
    py_error("int_length");
  return n;
}
Number* PyNumber::vbinomial(Number* n)
{ return n->binomial(this); 
}

Number* PyNumber::binomial(PyNumber *n)
{ PyObject *ans = py_funcs.py_binomial(n->pyobj,pyobj);
  if (!ans) throw std::runtime_error("pynumber::binomial");
  return new PyNumber(ans);
}

Number* PyNumber::bernoulli()
{ PyObject* ans = py_funcs.py_bernoulli(pyobj);
  if (!ans) throw std::runtime_error("pynumber::binomial");
  return new PyNumber(ans);
}
Number* PyNumber::vgcd(Number* n)
{ return n->gcd(this); 
}
Number* PyNumber::gcd(PyNumber *n)
{ PyObject *ans = py_funcs.py_gcd(n->pyobj,pyobj);
  if (!ans) throw std::runtime_error("pynumber::gcd");
  return new PyNumber(ans);
}

Number* PyNumber::vlcm(Number* n)
{ return n->lcm(this); 
}
Number* PyNumber::lcm(PyNumber *n)
{ PyObject *ans = py_funcs.py_lcm(n->pyobj,pyobj);
  if (!ans) throw std::runtime_error("pynumber::lcm");
  return new PyNumber(ans);
}



//////////////////////////////////////////////////////////////////////////////_
// class Number_T
///////////////////////////////////////////////////////////////////////////////
std::ostream& operator << (std::ostream& os, const Number_T& s) {
  PyObject* o;
    switch(s.t) {
    case LONG:
      return os << s.v._long;
    case DOUBLE:
      return os << s.v._double;
    case PYOBJECT:
      // TODO: maybe program around Python's braindead L suffix? PyLong_Check(s.v._pyobject) 
      o = PyObject_Repr(s.v._pyobject);
      if (!o) {
        PyErr_Clear();
	throw(std::runtime_error("operator<<(ostream, Number_T): exception printing python object"));
      } else {
	os << PyString_AsString(o);
	Py_DECREF(o);
      }
      return os;
    default:
      stub("operator <<: type not yet handled");
    }
  }
  PyObject* Number_T_to_pyobject(const Number_T& x) {
    // Returns a New Reference
    PyObject* o;
    switch(x.t) {
      case LONG:
	if (!(o = PyInt_FromLong(x.v._long))) {
	  py_error("Error coercing a long to an Integer");
	}
	return o;
      case DOUBLE:
	if (!(o =  PyFloat_FromDouble(x.v._double)))
	  py_error("Error creating double");
	return o;
	//if (!(o = PyObject_CallFunction(pyfunc_Float, "d", x.v._double))) 
	//  py_error("Error coercing a long to an Integer");

      case PYOBJECT:
        Py_INCREF(x.v._pyobject);
        return x.v._pyobject;

      default:
	stub("Number_T_to_pyobject -- not able to do conversion to pyobject; everything else will be nonsense");
	return 0;
      }
  }
  Number_T::Number_T(const char* s) { 
    // We should never use this. 
    verbose("Number_T(const char* s)");
    t = DOUBLE;
    sscanf(s, "%lf", &v._double);
  }

  Number_T::Number_T(PyObject* o) {
    verbose("Number_T::Number_T(PyObject* o)");
    if(! o) py_error("Error");
    t = PYOBJECT;
    v._pyobject = o; // STEAL a reference
  }

// Archive
Number_T::Number_T(const archive_node &n, lst &sym_lst)
{
	// get type information
	unsigned int t_tmp;
	if(!n.find_unsigned(std::string("T"), t_tmp))
		throw std::runtime_error("archive error: cannot read type info");
	t = Type(t_tmp);
	std::string str;
	PyObject *arg;
	switch(t) {
		case PYOBJECT:
			// read pickled python object to a string
			if(!n.find_string("S", str))
			    throw(std::runtime_error("archive error: cannot read pyobject data"));
			arg = Py_BuildValue("s#",str.c_str(), str.size());
			// unpickle
			v._pyobject = py_funcs.py_loads(arg);
			Py_DECREF(arg);
			if (PyErr_Occurred()) {
			    throw(std::runtime_error("archive error: caught exception in py_loads"));
			}
			Py_INCREF(v._pyobject);
			return;
		default:
			stub("unarchiving Number_T");
			return;
	}
}

void Number_T::archive(archive_node &n) const {
	// store type information
	n.add_unsigned("T", t);

	// create a string representation of this object
	std::string *tstr;
	switch(t) {
		case PYOBJECT:
			tstr = py_funcs.py_dumps(v._pyobject);
			if (PyErr_Occurred()) {
			    throw(std::runtime_error("archive error: exception in py_dumps"));
			}
			break;
		default:
			stub("archive Number_T");
	}

	n.add_string("S",*tstr);
	delete tstr;
}
  
  Number_T::~Number_T() {
    switch(t) {
    case PYOBJECT:
      Py_DECREF(v._pyobject);
      return;
    }
  }
    
  Number_T& Number_T::operator=(const Number_T& x) { 
    verbose("operator=");
    switch(x.t) {
    case DOUBLE:
      v._double = x.v._double; 
      break;

    case LONG:
      v._long = x.v._long; 
      break;

    case PYOBJECT:
      if (t == PYOBJECT) {
	Py_DECREF(v._pyobject);
      }
      Py_INCREF(x.v._pyobject);
      v._pyobject = x.v._pyobject; 
      break;
      
    default:
      stub("invalid type: operator= -- not able to do conversion! now total nonsense");
      break;
    };
    t = x.t;
    return *this; 
  }
  
  Number_T Number_T::operator-() { 
    verbose("operator-");
    switch(t) {
    case DOUBLE:
      return -v._double; 
    case LONG:
      return -v._long;
    case PYOBJECT:
      return PyNumber_Negative(v._pyobject);
    default:
      stub("invalid type: operator-() type not handled");
    }
  }

  Number_T::operator double() const { 
    verbose("operator double");
    double d;
    switch(t) {
    case DOUBLE:
      return v._double; 
    case LONG:
      return (double) v._long;
    case PYOBJECT:
      d = PyFloat_AsDouble(v._pyobject);
      if (d == -1 && PyErr_Occurred())
	py_error("Error converting to a double.");
      return d;
    default:
      std::cerr << "type = " << t << std::endl;
      stub("invalid type: operator double() type not handled");
    }
  }

  Number_T::operator int() const { 
    verbose("operator int");
    long n;
    switch(t) {
    case DOUBLE:
      return (int) v._double; 
    case LONG:
      todo("Need to check for overflow in   Number_T::operator int() const");
      return (int) v._long;
    case PYOBJECT:
      n = PyInt_AsLong(v._pyobject);
      if (((int)n != n) || (n == -1 && PyErr_Occurred()))
	py_error("Error converting to a long.");
      return n;
    default:
      stub("invalid type: operator int() type not handled");
    }
  }

  Number_T::operator long int() const { 
    verbose("operator long int");
    long int n;
    switch(t) {
    case DOUBLE:
      return (long int) v._double; 
    case LONG:
      return (long int) v._long;
    case PYOBJECT:
      n = PyInt_AsLong(v._pyobject);
      if (n == -1 && PyErr_Occurred()) {
	PyErr_Print();
	py_error("Overfloat converting to long int");
      }
      return n;
    default:
      stub("invalid type: operator long int() type not handled");
    }
  }

  unsigned Number_T::hash() const { 
    //verbose("hash");
    long res;
    switch(t) {
    case DOUBLE:
      return (unsigned int) v._double; 
    case LONG:
      return (unsigned int) v._long;
    case PYOBJECT:
      res = PyObject_Hash(v._pyobject);
      if (res == -1 && PyErr_Occurred()) {
	      throw(std::runtime_error("Number_T::hash() python function (__hash__) raised exception"));
      }
      return res;
    default:
      stub("invalid type: ::hash() type not handled");
    }
    //    return static_cast<unsigned>(v._double); 
  }
  
  bool Number_T::operator==(const Number_T& right) const { 
    verbose3("operator==", *this, right);
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a == b;
    }
    switch(t) {
    case DOUBLE:
      return v._double == right.v._double;
    case LONG:
      return v._long == right.v._long;
    case PYOBJECT:
      return py_funcs.py_is_equal(v._pyobject, right.v._pyobject);
    default:
      stub("invalid type: operator== type not handled");
    }
  }

  bool Number_T::operator!=(const Number_T& right) const { 
    verbose("operator!=");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a != b;
    }
    switch(t) {
    case DOUBLE:
      return v._double != right.v._double;
    case LONG:
      return v._long != right.v._long;
    case PYOBJECT:
      return (!py_funcs.py_is_equal(v._pyobject, right.v._pyobject));
    default:
      stub("invalid type: operator!= type not handled");
    }
  }

  int Number_T::compare_same_type(const Number_T& right) const { 
    verbose("compare_same_type");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a.compare_same_type(b);
    }
    switch(t) {
    case DOUBLE:
      return (v._double < right.v._double)?-1:(v._double > right.v._double);
    case LONG:
      return (v._long < right.v._long)?-1:(v._long > right.v._long);
    case PYOBJECT:
      int result;
      if (PyObject_Cmp(v._pyobject, right.v._pyobject, &result) == -1) {
	py_error("compare_same_type");
      }
      return result;
    default:
      stub("invalid type: compare_same_type type not handled");
    }

  }

  bool Number_T::operator<=(const Number_T& right) const { 
    verbose("operator<=");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a <= b;
    }
    switch(t) {
    case DOUBLE:
      return v._double <= right.v._double;
    case LONG:
      return v._long <= right.v._long;
    case PYOBJECT:
      int result;
      if (PyObject_Cmp(v._pyobject, right.v._pyobject, &result) == -1) {
	py_error("<=");
      }
      return (result <= 0);
    default:
      stub("invalid type: operator<= type not handled");
    }

  }
  
  bool Number_T::operator>=(const Number_T& right) const { 
    verbose("operator>=");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a >= b;
    }
    switch(t) {
    case DOUBLE:
      return v._double >= right.v._double;
    case LONG:
      return v._long >= right.v._long;
    case PYOBJECT:
      int result;
      if (PyObject_Cmp(v._pyobject, right.v._pyobject, &result) == -1) {
	py_error(">=");
      }
      return (result >= 0);
    default:
      stub("invalid type: operator!= type not handled");
    }
  }

  bool Number_T::operator<(const Number_T& right) const {
    verbose("operator<");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a < b;
    }

    switch(t) {
    case DOUBLE:
      return v._double < right.v._double;
    case LONG:
      return v._long < right.v._long;
    case PYOBJECT:
      int result;
      if (PyObject_Cmp(v._pyobject, right.v._pyobject, &result) == -1) {
	py_error("<");
      }
      return (result < 0);
    default:
      stub("invalid type: operator< type not handled");
    }
  }

  bool Number_T::operator>(const Number_T& right) const { 
    verbose("operator>");
    if (t != right.t) {
      Number_T a, b;
      coerce(a, b, *this, right);
      return a > b;
    }
    switch(t) {
    case DOUBLE:
      return v._double > right.v._double;
    case LONG:
      return v._long > right.v._long;
    case PYOBJECT:
      int result;
      if (PyObject_Cmp(v._pyobject, right.v._pyobject, &result) == -1) {
	py_error(">");
      }
      return (result > 0);
    default:
      stub("invalid type: operator> type not handled");
    }
  }
  
  /** Return the complex half-plane (left or right) in which the number lies.
   *  csgn(x)==0 for x==0, csgn(x)==1 for Re(x)>0 or Re(x)=0 and Im(x)>0,
   *  csgn(x)==-1 for Re(x)<0 or Re(x)=0 and Im(x)<0.
   *  */
  int Number_T::csgn() const { 
    verbose("csgn");
    switch(t) {
    case DOUBLE:
      if (v._double<0) 
	return -1; 
      if (v._double==0) 
	return 0; 
      return 1;
    case LONG:
      if (v._long<0) 
	return -1; 
      if (v._long==0) 
	return 0; 
      return 1;
    case PYOBJECT:
      int result;
      if (is_real()) {
	      if (PyObject_Cmp(v._pyobject, ZERO, &result) == -1)
		      py_error("csgn");
      } else {
	      PyObject *t = py_funcs.py_real(v._pyobject);
	      if (PyObject_Cmp(t, ZERO, &result) == -1)
		      py_error("csgn");
	      if (result == 0) {
		      Py_DECREF(t);
		      t = py_funcs.py_imag(v._pyobject);
		      if (PyObject_Cmp(t, ZERO, &result) == -1)
			      py_error("csgn");
		      Py_DECREF(t);
	      }
      }
      return result;
    default:
      stub("invalid type: csgn() type not handled");
    }
  }

  bool Number_T::is_zero() const { 
    verbose("is_zero");
    int a;
    switch(t) {
    case DOUBLE:
      return v._double == 0; 
    case LONG:
      return v._long == 0; 
    case PYOBJECT:
      a = PyObject_Not(v._pyobject);
      if (a==-1)
	py_error("is_zero");
      return a;
    default:
      std::cerr << "type = " << t << "\n";
      stub("invalid type: is_zero() type not handled");
    }
  }

  bool Number_T::is_positive() const { 
    verbose("is_positive");
    bool n;
    switch(t) {
    case DOUBLE:
      return v._double > 0; 
    case LONG:
      return v._long > 0; 
    case PYOBJECT:
      n = is_real() && (PyObject_Compare(v._pyobject, ZERO) > 0);
      if (PyErr_Occurred()) 
	py_error("is_positive");
      return n;
    default:
      stub("invalid type: is_positive() type not handled");
    }
  }

  bool Number_T::is_negative() const { 
    verbose("is_negative");
    bool n;
    switch(t) {
    case DOUBLE:
      return v._double < 0; 
    case LONG:
      return v._long < 0; 
    case PYOBJECT:
      n = is_real() && (PyObject_Compare(v._pyobject, ZERO) < 0);
      if (PyErr_Occurred()) 
	py_error("is_negative");
      return n;
    default:
      stub("invalid type: is_negative() type not handled");
    }
  }

  bool Number_T::is_integer() const { 
    verbose2("is_integer", *this);

    bool ans;
    PyObject* o;

    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return true;
    case PYOBJECT:
      return py_funcs.py_is_integer(v._pyobject);
    default:
      stub("invalid type: is_integer() type not handled");
    }
  }

  bool Number_T::is_cinteger() const { 
    verbose("is_crational");
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return true;
    case PYOBJECT:
      return py_funcs.py_is_cinteger(v._pyobject);
    default:
      stub("invalid type -- is_cinteger() type not handled");
    }
  }

  bool Number_T::is_pos_integer() const { 
    verbose("is_pos_integer");
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return (v._long > 0);
    case PYOBJECT:
      return (is_integer() && is_positive());
    default:
      stub("invalid type: is_pos_integer() type not handled");
    }
  }

  bool Number_T::is_nonneg_integer() const { 
    verbose("is_nonneg_integer");
    bool n;
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return (v._long >= 0);
    case PYOBJECT:
      n = (is_integer() && (PyObject_Compare(v._pyobject, ZERO) >= 0));
      if (PyErr_Occurred()) 
	py_error("is_nonneg_integer");
      return n;
    default:
      stub("invalid type: is_nonneg_integer() type not handled");
    }
  }
  
  bool Number_T::is_even() const { 
    verbose("is_even");

    bool ans;
    PyObject* o;

    if (!is_integer()) 
      return false;

    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return (v._long %2 == 0);
    case PYOBJECT:
      return py_funcs.py_is_even(v._pyobject);
    default:
      stub("invalid type: is_even() type not handled");
    }
  }

  bool Number_T::is_odd() const { 
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return (v._long %2 == 1);
    case PYOBJECT:
      return !is_even();
    default:
      stub("invalid type: is_odd() type not handled");
    }
  }
  
  bool Number_T::is_prime() const { 
    verbose("is_prime");
    PyObject* a;
    bool b;
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      a = Number_T_to_pyobject(*this);
      b = py_funcs.py_is_prime(a);
      Py_DECREF(a);
      return b;
    case PYOBJECT:
      return py_funcs.py_is_prime(v._pyobject);
    default:
      stub("invalid type: is_prime() type not handled");
    }
  }
   
  bool Number_T::is_rational() const { 
    verbose("is_rational");
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return true;
    case PYOBJECT:
      return py_funcs.py_is_rational(v._pyobject);
    default:
      stub("invalid type -- is_rational() type not handled");
    }
  }

  bool Number_T::is_crational() const { 
    verbose("is_crational");
    switch(t) {
    case DOUBLE:
      return false;
    case LONG:
      return true;
    case PYOBJECT:
      return py_funcs.py_is_crational(v._pyobject);
    default:
      stub("invalid type -- is_crational() type not handled");
    }
  }

  bool Number_T::is_real() const { 
    verbose("is_real");
    switch(t) {
    case DOUBLE:
    case LONG:
      return true;
    case PYOBJECT:
      return py_funcs.py_is_real(v._pyobject);
    default:
      stub("invalid type -- is_real() type not handled");
    }
  }

  int Number_T::get_parent_char() const {
    verbose("is_parent_pos_char");
    switch(t) {
    case DOUBLE:
      return 0;
    case LONG:
      return 0;
    case PYOBJECT:
      {
	int c = py_funcs.py_get_parent_char(v._pyobject);
	if (c == -1) py_error("error in py_get_parent_char");
	return c;
      }
    default:
      stub("invalid type -- is_parent_pos_char() type not handled");
    }
  }

  Number_T Number_T::numer() const { 
    verbose2("numer -- in:", *this);
    Number_T ans;
    PyObject* a;

    switch(t) {

    case DOUBLE:
    case LONG:
      ans = *this;
      break;

    case PYOBJECT:
      a = py_funcs.py_numer(v._pyobject);
      if (!a) py_error("numer");
      ans = a;
      break;
    default:
      stub("invalid type -- numer() type not handled");
      ans = *this;
    }
    verbose2("numer -- out:", ans);
    return ans;
  }

  Number_T Number_T::denom() const { 
    verbose2("denom -- in:", *this);
    Number_T ans;
    PyObject* a;

    switch(t) {
    case DOUBLE:
    case LONG:
      ans = 1;
      break;

    case PYOBJECT:
      a = py_funcs.py_denom(v._pyobject);
      if (!a) py_error("denom");
      ans = a;
      break;

    default:
      stub("invalid type -- denom() type not handled");
      ans = ONE;
    }
    verbose2("denom -- out:", ans);
    return ans;
  }

  Number_T Number_T::conjugate() const {
    PY_RETURN(py_funcs.py_conjugate);
  }
  
  Number_T Number_T::evalf(PyObject* parent) const {
	  PyObject *a = Number_T_to_pyobject(*this);
	  PyObject *ans = py_funcs.py_float(a, parent);
	  Py_DECREF(a);
	  if (!ans)
		  throw(std::runtime_error("numeric::evalf(): error calling py_float()"));

	  return ans; 
  }

  Number_T Number_T::step() const {
    PY_RETURN(py_funcs.py_step);
  }

  Number_T Number_T::fibonacci() const {
    PY_RETURN(py_funcs.py_fibonacci);
  }

  Number_T Number_T::sin() const {
    PY_RETURN(py_funcs.py_sin);
  }

  Number_T Number_T::cos() const {
    PY_RETURN(py_funcs.py_cos);
  }

  Number_T Number_T::zeta() const {
    PY_RETURN(py_funcs.py_zeta);
  }

  Number_T Number_T::exp() const {
    PY_RETURN(py_funcs.py_exp);
  }
  
  Number_T Number_T::log() const {
    PY_RETURN(py_funcs.py_log);
  }
  
  Number_T Number_T::tan() const {
    PY_RETURN(py_funcs.py_tan);
  }
  
  Number_T Number_T::asin() const {
    
    PY_RETURN(py_funcs.py_asin);
  }
    
  Number_T Number_T::acos() const {
    PY_RETURN(py_funcs.py_acos);
  }

  Number_T Number_T::atan() const {
    PY_RETURN(py_funcs.py_atan);
  }

  Number_T Number_T::atan(const Number_T& y) const {
    PY_RETURN2(py_funcs.py_atan2, y);
  }
  
  Number_T Number_T::sinh() const {
    PY_RETURN(py_funcs.py_sinh);
  }

  Number_T Number_T::cosh() const {
    PY_RETURN(py_funcs.py_cosh);
  }

  Number_T Number_T::tanh() const {
    PY_RETURN(py_funcs.py_tanh);
  }

  Number_T Number_T::asinh() const {
    PY_RETURN(py_funcs.py_asinh);
  }

  Number_T Number_T::acosh() const {
    PY_RETURN(py_funcs.py_acosh);
  }

  Number_T Number_T::atanh() const {
    PY_RETURN(py_funcs.py_atanh);
  }

/*  Number_T Number_T::Li(const Number_T &n, PyObject* parent) const {
    PyObject *aa = Number_T_to_pyobject(*this);	
    PyObject* nn = Number_T_to_pyobject(n);	
    PyObject *ans = py_funcs.py_li(aa, nn, parent);             
    if (!ans) py_error("error calling function");
    Py_DECREF(aa); Py_DECREF(nn);
    return ans; 
  }
*/
  Number_T Number_T::Li2() const {
    PY_RETURN(py_funcs.py_li2);
  }

  Number_T Number_T::lgamma() const {
    PY_RETURN(py_funcs.py_lgamma);
  }

  Number_T Number_T::tgamma() const {
    PY_RETURN(py_funcs.py_tgamma);
  }
  
  Number_T Number_T::psi() const {
    PY_RETURN(py_funcs.py_psi);
  }

  Number_T Number_T::psi(const Number_T& y) const {
    PY_RETURN2(py_funcs.py_psi2, y);
  }

  Number_T Number_T::factorial() const {
    PY_RETURN(py_funcs.py_factorial);
  }
  
  Number_T Number_T::doublefactorial() const {
    PY_RETURN(py_funcs.py_doublefactorial);
  }

  Number_T Number_T::isqrt() const {
    PY_RETURN(py_funcs.py_isqrt);
  }

  Number_T Number_T::sqrt() const {
    PY_RETURN(py_funcs.py_sqrt);
  }
  
  Number_T Number_T::abs() const {
    PY_RETURN(py_funcs.py_abs);
  }

  Number_T Number_T::mod(const Number_T &b) const {
    PY_RETURN2(py_funcs.py_mod, b);
  }

  Number_T Number_T::smod(const Number_T &b) const {
    PY_RETURN2(py_funcs.py_smod, b);
  }

  Number_T Number_T::irem(const Number_T &b) const {
    PY_RETURN2(py_funcs.py_irem, b);
  }

  Number_T Number_T::iquo(const Number_T &b) const {
    PY_RETURN2(py_funcs.py_iquo, b);
  }
  
  Number_T Number_T::iquo(const Number_T &b, Number_T& r) const {
    PY_RETURN3(py_funcs.py_iquo2, b, r);
  }

  int Number_T::int_length() const {
    PyObject* a = Number_T_to_pyobject(*this);
    int n = py_funcs.py_int_length(a);
    Py_DECREF(a);
    if (n == -1)
      py_error("int_length");
    return n;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // class numeric
  ///////////////////////////////////////////////////////////////////////////////

  GINAC_IMPLEMENT_REGISTERED_CLASS_OPT(numeric, basic,
				       print_func<print_context>(&numeric::do_print).
				       print_func<print_latex>(&numeric::do_print_latex).
				       print_func<print_csrc>(&numeric::do_print_csrc).
				       print_func<print_tree>(&numeric::do_print_tree).
				       print_func<print_python_repr>(&numeric::do_print_python_repr))


  //////////
  // default constructor
  //////////

  /** default constructor. Numerically it initializes to an integer zero. */
  numeric::numeric() : basic(&numeric::tinfo_static) 
  { 
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(0);   
  }

  //////////
  // other constructors
  //////////

  // public

  numeric::numeric(PyObject* o) : basic(&numeric::tinfo_static) 
  { 
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(o);
  }

  numeric::numeric(int i) : basic(&numeric::tinfo_static) 
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(i);
  }


  numeric::numeric(unsigned int i) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(i);
  }


  numeric::numeric(long i) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(i);
  }


  numeric::numeric(unsigned long i) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(i);
  }


  /** Constructor for rational numerics a/b.
   *
   *  @exception overflow_error (division by zero) */
  numeric::numeric(long numer, long denom) : basic(&numeric::tinfo_static)
  {
    if (!denom)
      throw std::overflow_error("numeric::div(): division by zero");
    //todo rational constructor
    number=(new PyNumber(numer))->vdiv(new PyNumber(denom));
    setflag(status_flags::evaluated | status_flags::expanded);
  }

  numeric::numeric(double d) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(d);
  }

  numeric::numeric(const char *s) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(s);
  }

  numeric::numeric(Number* n) : basic(&numeric::tinfo_static)
  { 
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(n);
  }
/*
  numeric::numeric(const GiNaC::numeric& n) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(n.number); //fix this
  }
  */


  /** constructor from C-style string.  It also accepts complex numbers in GiNaC
   *  notation like "2+5*I". */
/*  numeric::numeric(const char *s) : basic(&numeric::tinfo_static)
  {
    setflag(status_flags::evaluated | status_flags::expanded);
    number=new PyNumber(s);
  }*/


  //////////
  // archiving
  //////////
  numeric::numeric(const archive_node &n, lst &sym_lst): inherited(n, sym_lst)
  { number=new PyNumber(n,sym_lst);
    setflag(status_flags::evaluated | status_flags::expanded);
  }
  void numeric::archive(archive_node &n) const
  {
    number->archive(n);  
    inherited::archive(n);
  }

  DEFAULT_UNARCHIVE(numeric)

  //////////
  // functions overriding virtual functions from base classes
  //////////

  template<typename T1, typename T2> 
  static inline bool coerce(T1& dst, const T2& arg);

  /** 
   * @brief Check if CLN integer can be converted into int
   *
   * @sa http://www.ginac.de/pipermail/cln-list/2006-October/000248.html
   */
  template<>
  inline bool coerce<int, Number_T>(int& dst, const Number_T& arg)
  {
    dst = (int) arg;
  }

  template<>
  inline bool coerce<unsigned int, Number_T>(unsigned int& dst, const Number_T& arg)
  {
    dst = (long int) arg;   // TODO: worry about long int to unsigned int!!
  }

  void numeric::print_numeric(const print_context & c, const char *par_open,
		  const char *par_close, const char *imag_sym, 
		  const char *mul_sym, unsigned level, bool latex=false) const
  {
	  std::string* out;
	  if (latex) {
		  out = py_funcs.py_latex(to_pyobject(),level);
	  } else {
    	          out = py_funcs.py_repr(to_pyobject(),level);
          }
          c.s<<*out;
	  delete out;
	  return;
  }

  void numeric::do_print(const print_context & c, unsigned level) const
  {
    print_numeric(c, "(", ")", "I", "*", level, false);
  }

  void numeric::do_print_latex(const print_latex & c, unsigned level) const
  {
    print_numeric(c, "{(", ")}", "i", " ", level, true);
  }

  void numeric::do_print_csrc(const print_csrc & c, unsigned level) const
  {
    // TODO: not really needed?
    stub("print_csrc");
  }


  void numeric::do_print_tree(const print_tree & c, unsigned level) const
  {
    c.s << std::string(level, ' ') << number->str()
	<< " (" << class_name() << ")" << " @" << this
	<< std::hex << ", hash=0x" << hashvalue << ", flags=0x" << flags << std::dec
	<< std::endl;
  }

  void numeric::do_print_python_repr(const print_python_repr & c, unsigned level) const
  {
    c.s << class_name() << "('";
    print_numeric(c, "(", ")", "I", "*", level);
    c.s << "')";
  }

  bool numeric::info(unsigned inf) const
  {
    switch (inf) {
    case info_flags::numeric:
    case info_flags::polynomial:
    case info_flags::rational_function:
    case info_flags::expanded:
      return true;
    case info_flags::real:
      return is_real();
    case info_flags::rational:
    case info_flags::rational_polynomial:
      return is_rational();
    case info_flags::crational:
    case info_flags::crational_polynomial:
      return is_crational();
    case info_flags::integer:
    case info_flags::integer_polynomial:
      return is_integer();
    case info_flags::cinteger:
    case info_flags::cinteger_polynomial:
      return is_cinteger();
    case info_flags::positive:
      return is_positive();
    case info_flags::negative:
      return is_negative();
    case info_flags::nonnegative:
      return is_zero() || is_positive();
    case info_flags::posint:
      return is_pos_integer();
    case info_flags::negint:
      return is_integer() && is_negative();
    case info_flags::nonnegint:
      return is_nonneg_integer();
    case info_flags::even:
      return is_even();
    case info_flags::odd:
      return is_odd();
    case info_flags::prime:
      return is_prime();
    case info_flags::algebraic:
      return !is_real();
    }
    return false;
  }

  bool numeric::is_polynomial(const ex & var) const
  {
    return true;
  }

  int numeric::degree(const ex & s) const
  {
    // In sage deg (0) != 0 !!!
    return 0;
  }

  int numeric::ldegree(const ex & s) const
  {
    return 0;
  }

  ex numeric::coeff(const ex & s, int n) const
  {
    return n==0 ? *this : _ex0;
  }

  /** Disassemble real part and imaginary part to scan for the occurrence of a
   *  single number.  Also handles the imaginary unit.  It ignores the sign on
   *  both this and the argument, which may lead to what might appear as funny
   *  results:  (2+I).has(-2) -> true.  But this is consistent, since we also
   *  would like to have (-2+I).has(2) -> true and we want to think about the
   *  sign as a multiplicative factor. */
  bool numeric::has(const ex &other, unsigned options) const
  {
    if (!is_exactly_a<numeric>(other))
      return false;
    const numeric &o = ex_to<numeric>(other);
    if (this->is_equal(o) || this->is_equal(-o))
      return true;
    if (o.imag().is_zero()) {   // e.g. scan for 3 in -3*I
      if (!this->real().is_equal(*_num0_p))
	if (this->real().is_equal(o) || this->real().is_equal(-o))
	  return true;
      if (!this->imag().is_equal(*_num0_p))
	if (this->imag().is_equal(o) || this->imag().is_equal(-o))
	  return true;
      return false;
    }
    else {
      if (o.is_equal(I))  // e.g scan for I in 42*I
	return !this->is_real();
      if (o.real().is_zero())  // e.g. scan for 2*I in 2*I+1
	if (!this->imag().is_equal(*_num0_p))
	  if (this->imag().is_equal(o*I) || this->imag().is_equal(-o*I))
	    return true;
    }
    return false;
  }


  /** Evaluation of numbers doesn't do anything at all. */
  ex numeric::eval(int level) const
  {
    // Warning: if this is ever gonna do something, the ex constructors from all kinds
    // of numbers should be checking for status_flags::evaluated.
    return this->hold();
  }


  /** Cast numeric into a floating-point object.  For example exact numeric(1) is
   *  returned as a 1.0000000000000000000000 and so on according to how Digits is
   *  currently set.  In case the object already was a floating point number the
   *  precision is trimmed to match the currently set default.
   *
   *  @param level  ignored, only needed for overriding basic::evalf.
   *  @return  an ex-handle to a numeric. */
  ex numeric::evalf(int level, PyObject* parent) const
  {
     return numeric(number->evalf(parent));
  }

  ex numeric::conjugate() const
  {
    return numeric(number->conjugate());
  }

  ex numeric::real_part() const
  {
    return real();
  }

  ex numeric::imag_part() const
  {
    return imag();
  }

  // protected

  /** This method establishes a canonical order on all numbers.  For complex
   *  numbers this is not possible in a mathematically consistent way but we need
   *  to establish some order and it ought to be fast.  So we simply define it
   *  to be compatible with our method csgn.
   *
   *  @return csgn(*this-other) */
  int numeric::compare_same_type(const basic &other) const
  {
    GINAC_ASSERT(is_exactly_a<numeric>(other));
    const numeric &o = static_cast<const numeric &>(other);

    return number->vcompare_same_type(o.number);
  }


  bool numeric::is_equal_same_type(const basic &other) const
  {
    GINAC_ASSERT(is_exactly_a<numeric>(other));
    const numeric &o = static_cast<const numeric &>(other);
	
    return this->is_equal(o);
  }


  unsigned numeric::calchash() const
  {
    return number->hash();
  }


  //////////
  // new virtual functions which can be overridden by derived classes
  //////////

  // none

  //////////
  // non-virtual functions in this class
  //////////

  // public

  /** Numerical addition method.  Adds argument to *this and returns result as
   *  a numeric object. */
  const numeric numeric::add(const numeric &other) const
  {
    return numeric(number->vadd(other.number));
  }


  /** Numerical subtraction method.  Subtracts argument from *this and returns
   *  result as a numeric object. */
  const numeric numeric::sub(const numeric &other) const
  {
    return numeric(number->vsub(other.number));
  }


  /** Numerical multiplication method.  Multiplies *this and argument and returns
   *  result as a numeric object. */
  const numeric numeric::mul(const numeric &other) const
  {
    return numeric(number->vmul(other.number));
  }


  /** Numerical division method.  Divides *this by argument and returns result as
   *  a numeric object.
   *
   *  @exception overflow_error (division by zero) */
  const numeric numeric::div(const numeric &other) const
  {
    //todo -- delete
    //
    // 
    if (other.is_zero()) 
      throw std::overflow_error("numeric::div(): division by zero");
    return numeric(number->vdiv(other.number));
  }


  /** Numerical exponentiation.  Raises *this to the power given as argument and
   *  returns result as a numeric object. */
  const numeric numeric::power(const numeric &other) const
  {
    
    //other.dbgprint();
    //this->dbgprint();
    return numeric(number->vpow(other.number));
  }

  /** Numerical addition method.  Adds argument to *this and returns result as
   *  a numeric object on the heap.  Use internally only for direct wrapping into
   *  an ex object, where the result would end up on the heap anyways. */
  const numeric &numeric::add_dyn(const numeric &other) const
  { 
    // Efficiency shortcut: trap the neutral element by pointer.  This hack
    // is supposed to keep the number of distinct numeric objects low.
    if (this==_num0_p)
      return other;
    else if (&other==_num0_p)
      return *this;
	
    return static_cast<const numeric &>((new numeric(number->vadd(other.number)))->
					setflag(status_flags::dynallocated));
  }


  /** Numerical subtraction method.  Subtracts argument from *this and returns
   *  result as a numeric object on the heap.  Use internally only for direct
   *  wrapping into an ex object, where the result would end up on the heap
   *  anyways. */
  const numeric &numeric::sub_dyn(const numeric &other) const
  {
    // Efficiency shortcut: trap the neutral exponent (first by pointer).  This
    // hack is supposed to keep the number of distinct numeric objects low.
    if (&other==_num0_p || (other.number->is_zero()))
      return *this;
	
    return static_cast<const numeric &>((new numeric(number->vsub(other.number)))->
					setflag(status_flags::dynallocated));
  }


  /** Numerical multiplication method.  Multiplies *this and argument and returns
   *  result as a numeric object on the heap.  Use internally only for direct
   *  wrapping into an ex object, where the result would end up on the heap
   *  anyways. */
  const numeric &numeric::mul_dyn(const numeric &other) const
  {
    // Efficiency shortcut: trap the neutral element by pointer.  This hack
    // is supposed to keep the number of distinct numeric objects low.
    if (this==_num1_p)
    { 
      return other;
    }
    else if (&other==_num1_p)
    { 
      return *this;
    }

    return static_cast<const numeric &>((new numeric(new PyNumber(number->vmul(other.number))))->
					setflag(status_flags::dynallocated)); //unmodify
  }


  /** Numerical division method.  Divides *this by argument and returns result as
   *  a numeric object on the heap.  Use internally only for direct wrapping
   *  into an ex object, where the result would end up on the heap
   *  anyways.
   *
   *  @exception overflow_error (division by zero) */
  const numeric &numeric::div_dyn(const numeric &other) const
  {
    // Efficiency shortcut: trap the neutral element by pointer.  This hack
    // is supposed to keep the number of distinct numeric objects low.
    if (&other==_num1_p)
      return *this;
    if (other.number->is_zero())
      throw std::overflow_error("division by zero");
    return static_cast<const numeric &>((new numeric(number->vdiv(other.number)))->
					setflag(status_flags::dynallocated));
  }


  /** Numerical exponentiation.  Raises *this to the power given as argument and
   *  returns result as a numeric object on the heap.  Use internally only for
   *  direct wrapping into an ex object, where the result would end up on the
   *  heap anyways. */
  const numeric &numeric::power_dyn(const numeric &other) const
  {
    // Efficiency shortcut: trap the neutral exponent (first try by pointer, then
    // try harder, since calls to cln::expt() below may return amazing results for
    // floating point exponent 1.0).
    if (&other==_num1_p || (other.number->vequal(_num1_p->number)))
      return *this;
	
    return static_cast<const numeric &>((new numeric(number->vpow(other.number)))->
					setflag(status_flags::dynallocated));
  }


  const numeric &numeric::operator=(int i)
  {
    return operator=(numeric(i));
  }


  const numeric &numeric::operator=(unsigned int i)
  {
    return operator=(numeric(i));
  }


  const numeric &numeric::operator=(long i)
  {
    return operator=(numeric(i));
  }


  const numeric &numeric::operator=(unsigned long i)
  {
    return operator=(numeric(i));
  }


  const numeric &numeric::operator=(double d)
  {
    return operator=(numeric(d));
  }


  const numeric &numeric::operator=(const char * s)
  {
    return operator=(numeric(s));
  }


  /** Inverse of a number. */
  const numeric numeric::inverse() const
  {
    if (number->is_zero())
      throw std::overflow_error("numeric::inverse(): division by zero");
    return (new PyNumber(ONE))->vdiv(number);
  }

  /** Return the step function of a numeric. The imaginary part of it is
   *  ignored because the step function is generally considered real but
   *  a numeric may develop a small imaginary part due to rounding errors.
   */
  numeric numeric::step() const
  {
    return number->step();
  }

  /** Return the complex half-plane (left or right) in which the number lies.
   *  csgn(x)==0 for x==0, csgn(x)==1 for Re(x)>0 or Re(x)=0 and Im(x)>0,
   *  csgn(x)==-1 for Re(x)<0 or Re(x)=0 and Im(x)<0.
   *
   *  @see numeric::compare(const numeric &other) */
  int numeric::csgn() const
  {
    return number->csgn();
  }

  int numeric::compare(const numeric &other) const
  {
    return (*this-other).csgn();
  }

  bool numeric::is_equal(const numeric &other) const
  {
    return number->vequal(other.number);
  }


  /** True if object is zero. */
  bool numeric::is_zero() const
  {
    return number->is_zero();
  }


  /** True if object is not complex and greater than zero. */
  bool numeric::is_positive() const
  {
    return number->is_positive();
  }


  /** True if object is not complex and less than zero. */
  bool numeric::is_negative() const
  {
    return number->is_negative();
  }


  /** True if object is a non-complex integer. */
  bool numeric::is_integer() const
  {
    return number->is_integer();
  }


  /** True if object is an exact integer greater than zero. */
  bool numeric::is_pos_integer() const
  {
    return number->is_pos_integer();
  }


  /** True if object is an exact integer greater or equal zero. */
  bool numeric::is_nonneg_integer() const
  {
    return number->is_nonneg_integer();
  }


  /** True if object is an exact even integer. */
  bool numeric::is_even() const
  {
    return number->is_even();
  }


  /** True if object is an exact odd integer. */
  bool numeric::is_odd() const
  {
    return number->is_odd();
  }


  /** Probabilistic primality test.
   *
   *  @return  true if object is exact integer and prime. */
  bool numeric::is_prime() const
  {
    return number->is_prime();
  }


  /** True if object is an exact rational number, may even be complex
   *  (denominator may be unity). */
  bool numeric::is_rational() const
  {
    return number->is_rational();
  }


  /** True if object is a real integer, rational or float (but not complex). */
  bool numeric::is_real() const
  {
    return number->is_real();
  }

  /** True if the parent of the object has positive characteristic. */
  bool numeric::is_parent_pos_char() const
  {
    return number->get_parent_char() > 0;
  }

  /** Returns the characteristic of the parent of this object. */
  int numeric::get_parent_char() const
  {
    return number->get_parent_char();
  }

  bool numeric::operator==(const numeric &other) const
  {
    return number->vequal(other.number);
  }

  bool numeric::operator!=(const numeric &other) const
  {
    return number->vnot_equal(other.number);
  }


  /** True if object is element of the domain of integers extended by I, i.e. is
   *  of the form a+b*I, where a and b are integers. */
  bool numeric::is_cinteger() const
  {
    return number->is_cinteger();
  }


  /** True if object is an exact rational number, may even be complex
   *  (denominator may be unity). */
  bool numeric::is_crational() const
  {
    return number->is_crational();
  }


  /** Numerical comparison: less.
   *
   *  @exception invalid_argument (complex inequality) */ 
  bool numeric::operator<(const numeric &other) const
  {
    return number->vless(other.number);
  }


  /** Numerical comparison: less or equal.
   *
   *  @exception invalid_argument (complex inequality) */ 
  bool numeric::operator<=(const numeric &other) const
  {
    return number->vless_equal(other.number);
  }


  /** Numerical comparison: greater.
   *
   *  @exception invalid_argument (complex inequality) */ 
  bool numeric::operator>(const numeric &other) const
  {
    return number->vgreater(other.number); 
  }


  /** Numerical comparison: greater or equal.
   *
   *  @exception invalid_argument (complex inequality) */  
  bool numeric::operator>=(const numeric &other) const
  {
    return number->vgreater_equal(other.number);
  }


  /** Converts numeric types to machine's int.  You should check with
   *  is_integer() if the number is really an integer before calling this method.
   *  You may also consider checking the range first. */
  int numeric::to_int() const //not ok
  {
    GINAC_ASSERT(this->is_integer());
    return number->to_int(); 
  }


  /** Converts numeric types to machine's long.  You should check with
   *  is_integer() if the number is really an integer before calling this method.
   *  You may also consider checking the range first. */
  long numeric::to_long() const //not ok
  {
    GINAC_ASSERT(this->is_integer());
    return number->to_long_int(); 
  }

  /* Return the underlying Python object corresponding to this
     numeric.  If this numeric isn't implemented using a Python
     object, the corresponding Python object is constructed on
     the fly.

     Returns a NEW REFERENCE.
  */
  PyObject* numeric::to_pyobject() const //not ok
  {
      return number->pyobject();
  }

  /** Converts numeric types to machine's double. You should check with is_real()
   *  if the number is really not complex before calling this method. */
  double numeric::to_double() const //not ok

  {
    GINAC_ASSERT(this->is_real());
    return number->to_double(); 
  }

  /** Real part of a number. */
  const numeric numeric::real() const
  { return numeric(number->real());
  }


  /** Imaginary part of a number. */
  const numeric numeric::imag() const
  { return numeric(number->imag());
  }


  /** Numerator.  Computes the numerator of rational numbers, rationalized
   *  numerator of complex if real and imaginary part are both rational numbers
   *  (i.e numer(4/3+5/6*I) == 8+5*I), the number carrying the sign in all other
   *  cases. */
  const numeric numeric::numer() const //not polymorphic
  { 
    return number->numer();
  }

  /** Denominator.  Computes the denominator of rational numbers, common integer
   *  denominator of complex if real and imaginary part are both rational numbers
   *  (i.e denom(4/3+5/6*I) == 6), one in all other cases. */
  const numeric numeric::denom() const //not polymorphic
  { 
    return number->denom();
  }


  /** Size in binary notation.  For integers, this is the smallest n >= 0 such
   *  that -2^n <= x < 2^n. If x > 0, this is the unique n > 0 such that
   *  2^(n-1) <= x < 2^n.
   *
   *  @return  number of bits (excluding sign) needed to represent that number
   *  in two's complement if it is an integer, 0 otherwise. */    
  int numeric::int_length() const
  {
    return number->int_length();
  }
  const numeric numeric::exp() const
  { return numeric(number->exp()); 
  } 
  const numeric numeric::log() const
  { 
    return numeric(number->log());
  }
  const numeric numeric::sin() const
  { 
    return numeric(number->sin());
  }
  const numeric numeric::cos() const
  { 
    return numeric(number->cos());
  }
  const numeric numeric::tan() const
  { 
    return numeric(number->tan());
  }
  const numeric numeric::asin() const
  {
    
    return numeric(number->asin());
  }
  const numeric numeric::acos() const
  { 
    return numeric(number->acos());
  }
  const numeric numeric::atan() const
  { 
    return numeric(number->atan());
  }
   const numeric numeric::atan(const numeric& n) const
  { 
    return numeric(number->vatan(n.number));
  }
  const numeric numeric::sinh() const
  {
    return numeric(number->sinh());
  }
  const numeric numeric::cosh() const
  {
    return numeric(number->cosh());
  }
  const numeric numeric::tanh() const
  {
    return numeric(number->tanh());
  }
  const numeric numeric::asinh() const
  {
    return numeric(number->asinh());
  }
  const numeric numeric::acosh() const
  {
    return numeric(number->acosh());
  }
  const numeric numeric::atanh() const
  {
    return numeric(number->atanh());
  }

 /* const numeric numeric::Li(PyObject* parent)
  {
    return numeric(number->Li(parent));
  }*/
  const numeric numeric::Li(const numeric &n, PyObject* parent) const
  {
    return numeric(number->vLi(n.number,parent));
  }

  const numeric numeric::Li2() const
  {
    return numeric(number->Li2());
  }
  const numeric numeric::zeta() const
  {
    return numeric(number->zeta());
  }
  const numeric numeric::lgamma() const
  {
    return numeric(number->lgamma());
  }
  const numeric numeric::tgamma() const
  { 
    return numeric(number->tgamma());
  }
  const numeric numeric::psi() const
  {
    return numeric(number->psi());
  }
  const numeric numeric::psi(const numeric &x) const
  {
    return numeric(number->vpsi(x.number));
  }
  const numeric numeric::factorial() const
  {
    return numeric(number->factorial());
  }
  const numeric numeric::doublefactorial() const
  {
    return numeric(number->doublefactorial());
  }
  const numeric numeric::binomial(const numeric &k)  const
  { return numeric(number->vbinomial(k.number));
  }
  const numeric numeric::bernoulli() const
  { return numeric(number->bernoulli());
  }
  const numeric numeric::fibonacci() const
  {
    return numeric(number->fibonacci());
  }
  const numeric numeric::isqrt() const
  {
    return numeric(number->isqrt());
  }
  const numeric numeric::sqrt() const
  {
    return numeric(number->sqrt());
  }
  const numeric numeric::abs() const
  {
    return numeric(number->abs());
  }
  numeric numeric::integer_content() const
  {
    return numeric(number->abs());
  }
  numeric numeric::max_coefficient() const
  {
    return numeric(number->abs());
  }
  const numeric numeric::mod(const numeric &b) const
  {
    return numeric(number->vmod(b.number));
  }
  ex numeric::smod(const numeric &b) const // inherited basic
  {
    return numeric(number->vsmod(b.number));
  }
  const numeric numeric::smod2(const numeric &b) const
  {
    return numeric(number->vsmod(b.number));
  }
  const numeric numeric::irem(const numeric &b) const
  {
    return numeric(number->virem(b.number));
  }
  const numeric numeric::iquo(const numeric &b) const
  {
    return numeric(number->viquo(b.number));
  }
  const numeric numeric::iquo(const numeric &b,const numeric &r) const
  {
    return numeric(number->viquo(b.number, r.number));
  }
  const numeric numeric::gcd(const numeric &b) const
  { return numeric(number->vgcd(b.number));
  }
  const numeric numeric::lcm(const numeric &b) const
  { return numeric(number->vlcm(b.number));
  }
  //////////
  // global constants
  //////////

  /** Imaginary unit.  This is not a constant but a numeric since we are
   *  natively handing complex numbers anyways, so in each expression containing
   *  an I it is automatically eval'ed away anyhow. */

//this should go away in the future
  /** Exponential function.
   *
   *  @return  arbitrary precision numerical exp(x). */
  const numeric exp(const numeric &x)
  {
    return x.exp();
  }


  /** Natural logarithm.
   *
   *  @param x complex number
   *  @return  arbitrary precision numerical log(x).
   *  @exception pole_error("log(): logarithmic pole",0) */
  const numeric log(const numeric &x)
  {
	  /*
    if (x.is_zero())
      throw pole_error("log(): logarithmic pole",0);
      */
    return x.log();
  }


  /** Numeric sine (trigonometric function).
   *
   *  @return  arbitrary precision numerical sin(x). */
  const numeric sin(const numeric &x)
  {
    return x.sin();
  }


  /** Numeric cosine (trigonometric function).
   *
   *  @return  arbitrary precision numerical cos(x). */
  const numeric cos(const numeric &x)
  {
    return x.cos();
  }


  /** Numeric tangent (trigonometric function).
   *
   *  @return  arbitrary precision numerical tan(x). */
  const numeric tan(const numeric &x)
  {
    return x.tan();
  }
	

  /** Numeric inverse sine (trigonometric function).
   *
   *  @return  arbitrary precision numerical asin(x). */
  const numeric asin(const numeric &x)
  {
    return x.asin();
  }


  /** Numeric inverse cosine (trigonometric function).
   *
   *  @return  arbitrary precision numerical acos(x). */
  const numeric acos(const numeric &x)
  {
    return x.acos();
  }
	

  /** Numeric arcustangent.
   *
   *  @param x complex number
   *  @return atan(x)
   *  @exception pole_error("atan(): logarithmic pole",0) if x==I or x==-I. */
  const numeric atan(const numeric &x)
  { return x.atan();
  }


  /** Numeric arcustangent of two arguments, analytically continued in a suitable way.
   *
   *  @param y complex number
   *  @param x complex number
   *  @return -I*log((x+I*y)/sqrt(x^2+y^2)), which is equal to atan(y/x) if y and
   *    x are both real.
   *  @exception pole_error("atan(): logarithmic pole",0) if y/x==+I or y/x==-I. */
  const numeric atan(const numeric &y, const numeric &x)
  {
    return x.atan(y);
  }


  /** Numeric hyperbolic sine (trigonometric function).
   *
   *  @return  arbitrary precision numerical sinh(x). */
  const numeric sinh(const numeric &x)
  {
    return x.sinh();
  }


  /** Numeric hyperbolic cosine (trigonometric function).
   *
   *  @return  arbitrary precision numerical cosh(x). */
  const numeric cosh(const numeric &x)
  {
    return x.cosh();
  }


  /** Numeric hyperbolic tangent (trigonometric function).
   *
   *  @return  arbitrary precision numerical tanh(x). */
  const numeric tanh(const numeric &x)
  {
    return x.tanh();
  }
	

  /** Numeric inverse hyperbolic sine (trigonometric function).
   *
   *  @return  arbitrary precision numerical asinh(x). */
  const numeric asinh(const numeric &x)
  {
    return x.asinh();
  }


  /** Numeric inverse hyperbolic cosine (trigonometric function).
   *
   *  @return  arbitrary precision numerical acosh(x). */
  const numeric acosh(const numeric &x)
  {
    return x.acosh();
  }


  /** Numeric inverse hyperbolic tangent (trigonometric function).
   *
   *  @return  arbitrary precision numerical atanh(x). */
  const numeric atanh(const numeric &x)
  {
    return x.atanh();
  }


  /** Numeric evaluation of Dilogarithm within circle of convergence (unit
   *  circle) using a power series. */

  /** Numeric evaluation of Dilogarithm.  The domain is the entire complex plane,
   *  the branch cut lies along the positive real axis, starting at 1 and
   *  continuous with quadrant IV.
   *
   *  @return  arbitrary precision numerical Li2(x). */
  const numeric Li2(const numeric &x)
  {
    return x.Li2();
  }


  /** Evaluation of Riemann's Zeta function.  */
  const numeric zeta(const numeric &x)
  {
    return x.zeta();
  }


  /** The Gamma function.
   *  Use the Lanczos approximation. If the coefficients used here are not
   *  sufficiently many or sufficiently accurate, more can be calculated
   *  using the program doc/examples/lanczos.cpp. In that case, be sure to
   *  read the comments in that file. */
  const numeric lgamma(const numeric &x)
  {
    return x.lgamma();
  }

  const numeric tgamma(const numeric &x)
  {
    return x.tgamma();
  }


  /** The psi function (aka polygamma function). */
  const numeric psi(const numeric &x)
  {
    return x.psi();
  }


  /** The psi functions (aka polygamma functions). */
  const numeric psi(const numeric &n, const numeric &x)
  {
    return n.psi(x);
  }


  /** Factorial combinatorial function.
   *
   *  @param n  integer argument >= 0
   *  @exception range_error (argument must be integer >= 0) */
  const numeric factorial(const numeric &n)
  {
    return n.factorial();
  }


  /** The double factorial combinatorial function.  (Scarcely used, but still
   *  useful in cases, like for exact results of tgamma(n+1/2) for instance.)
   *
   *  @param n  integer argument >= -1
   *  @return n!! == n * (n-2) * (n-4) * ... * ({1|2}) with 0!! == (-1)!! == 1
   *  @exception range_error (argument must be integer >= -1) */
  const numeric doublefactorial(const numeric &n)
  {
    return n.doublefactorial();
  }


  /** The Binomial coefficients.  It computes the binomial coefficients.  For
   *  integer n and k and positive n this is the number of ways of choosing k
   *  objects from n distinct objects.  If n is negative, the formula
   *  binomial(n,k) == (-1)^k*binomial(k-n-1,k) is used to compute the result. */
  const numeric binomial(const numeric &n, const numeric &k) {
    return n.binomial(k);
  }


  /** Bernoulli number.  The nth Bernoulli number is the coefficient of x^n/n!
   *  in the expansion of the function x/(e^x-1).
   *
   *  @return the nth Bernoulli number (a rational number).
   *  @exception range_error (argument must be integer >= 0) */
  const numeric bernoulli(const numeric &n)
  {
    return n.bernoulli();
  }


  /** Fibonacci number.  The nth Fibonacci number F(n) is defined by the
   *  recurrence formula F(n)==F(n-1)+F(n-2) with F(0)==0 and F(1)==1.
   *
   *  @param n an integer
   *  @return the nth Fibonacci number F(n) (an integer number)
   *  @exception range_error (argument must be an integer) */
  const numeric fibonacci(const numeric &n)
  {
    return n.fibonacci();
  }


  const numeric abs(const numeric& x)
  {
    return x.abs();
  }


  /** Modulus (in positive representation).
   *  In general, mod(a,b) has the sign of b or is zero, and rem(a,b) has the
   *  sign of a or is zero. This is different from Maple's modp, where the sign
   *  of b is ignored. It is in agreement with Mathematica's Mod.
   *
   *  @return a mod b in the range [0,abs(b)-1] with sign of b if both are
   *  integer, 0 otherwise. */
  const numeric mod(const numeric &a, const numeric &b)
  {
    return a.mod(b);
  }


  /** Modulus (in symmetric representation).
   *  Equivalent to Maple's mods.
   *
   *  @return a mod b in the range [-iquo(abs(b)-1,2), iquo(abs(b),2)]. */
  const numeric smod(const numeric &a, const numeric &b)
  {
    return a.smod2(b);
  }


  /** Numeric integer remainder.
   *  Equivalent to Maple's irem(a,b) as far as sign conventions are concerned.
   *  In general, mod(a,b) has the sign of b or is zero, and irem(a,b) has the
   *  sign of a or is zero.
   *
   *  @return remainder of a/b if both are integer, 0 otherwise.
   *  @exception overflow_error (division by zero) if b is zero. */
  const numeric irem(const numeric &a, const numeric &b)
  {
    return a.irem(b);
  }

  /** Numeric integer quotient.
   *  Equivalent to Maple's iquo as far as sign conventions are concerned.
   *  
   *  @return truncated quotient of a/b if both are integer, 0 otherwise.
   *  @exception overflow_error (division by zero) if b is zero. */
  const numeric iquo(const numeric &a, const numeric &b)
  {
    return a.iquo(b);
  }


  /** Numeric integer quotient.
   *  Equivalent to Maple's iquo(a,b,'r') it obeyes the relation
   *  r == a - iquo(a,b,r)*b.
   *
   *  @return truncated quotient of a/b and remainder stored in r if both are
   *  integer, 0 otherwise.
   *  @exception overflow_error (division by zero) if b is zero. */
  const numeric iquo(const numeric &a, const numeric &b, numeric &r)
  {
    return a.iquo(b, r);
  }


  /** Greatest Common Divisor.
   *   
   *  @return  The GCD of two numbers if both are integer, a numerical 1
   *  if they are not. */
  const numeric gcd(const numeric &a, const numeric &b)
  { return a.gcd(b);
  }


  /** Least Common Multiple.
   *   
   *  @return  The LCM of two numbers if both are integer, the product of those
   *  two numbers if they are not. */
  const numeric lcm(const numeric &a, const numeric &b)
  { return a.lcm(b);
  }


  /** Numeric square root.
   *  If possible, sqrt(x) should respect squares of exact numbers, i.e. sqrt(4)
   *  should return integer 2.
   *
   *  @param x numeric argument
   *  @return square root of x. Branch cut along negative real axis, the negative
   *  real axis itself where imag(x)==0 and real(x)<0 belongs to the upper part
   *  where imag(x)>0. */
  const numeric sqrt(const numeric &x)
  {
    return x.sqrt();
  }


  /** Integer numeric square root. */
  const numeric isqrt(const numeric &x)
  {
    return x.isqrt();
  }

  /** Floating point evaluation of Sage's constants. */
  ex ConstantEvalf(unsigned serial, PyObject* parent)
  { 
    PyObject* x = py_funcs.py_eval_constant(serial, parent);
    if (!x) py_error("error getting digits of constant");
    return x;
  }

    ex UnsignedInfinityEvalf(unsigned serial, PyObject* parent)
	{
		PyObject* x = py_funcs.py_eval_unsigned_infinity();
		return x;
	}

    ex InfinityEvalf(unsigned serial, PyObject* parent)
	{
		PyObject* x = py_funcs.py_eval_infinity();
		return x;
	}

    ex NegInfinityEvalf(unsigned serial, PyObject* parent)
	{
		PyObject* x = py_funcs.py_eval_neg_infinity();
		return x;
	}


  Number_T::Number_T()  { 
    verbose("Number_T::Number_T()");

    //v._pyobject = Integer(0);
    //t = PYOBJECT;
    // t = LONG;
    // v._long = 0;

    t = PYOBJECT;
    if (!(v._pyobject = PyInt_FromLong(0)))
      py_error("Error creating 0 number");

    // v._pyobject = Integer_Zero;
    // Py_INCREF(v._pyobject);
  }

  Number_T::Number_T(const int& x) { 
    verbose("Number_T::Number_T(const int& x)");
    //v._pyobject = Integer(x);
    //t = PYOBJECT;

    //if (!(v._pyobject = PyObject_CallFunction(pyfunc_Integer, "i", x)))
    
    t = PYOBJECT;
    if (!(v._pyobject = PyInt_FromLong(x)))
      py_error("Error creating int");

    //t = LONG;
    // v._long = x;
  }
  Number_T::Number_T(const long int& x) { 
    verbose("Number_T::Number_T(const long int& x)");
    t = PYOBJECT;
    if (!(v._pyobject = PyInt_FromLong(x)))
      py_error("Error creating int");
    //t = LONG;
    //v._long = x;
    //v._pyobject = Integer(x);
    //t = PYOBJECT;
  }

  Number_T::Number_T(const unsigned int& x) { 
    // TODO !!!! -- these won't work since Integer assumes
    // input is signed!!!
    verbose("Number_T::Number_T(const unsigned int& x)");
    v._pyobject = Integer(x);
    t = PYOBJECT;
  }

  Number_T::Number_T(const unsigned long& x) { 
    verbose("Number_T::Number_T(const unsigned long& x)");
    t = PYOBJECT;
    v._pyobject = Integer(x);
  }

  Number_T::Number_T(const double& x) { 
    verbose("Number_T::Number_T(const double& x)");

    t = PYOBJECT;
    if (!(v._pyobject =  PyFloat_FromDouble(x)))
      py_error("Error creating double");

    //    if (!(v._pyobject = py_funcs.py_RDF_from_double(x)))


    //t = DOUBLE;
    //v._double = x; 

    //if (!(v._pyobject = PyObject_CallFunction(pyfunc_Float, "d", x)))

    //if (!(v._pyobject =  PyFloat_FromDouble(x)))
    //py_error("Error creating double");

    //t = PYOBJECT;
    //v._pyobject = PyFloat_FromDouble(x);
  }

  Number_T::Number_T(const Number_T& x) { 
    verbose("Number_T::Number_T(const Number_T& x)");
    t = x.t;
    switch(t) {
    case DOUBLE:
      v._double = x.v._double;
      return;
    case LONG:
      v._long = x.v._long;
      return;
    case PYOBJECT:
      v._pyobject = x.v._pyobject;
      Py_INCREF(v._pyobject);
      return;
    default:
      std::cerr << "type = " << t << "\n";
      stub("invalid type: Number_T(const Number_T& x) type not handled");
    }
  }

  void coerce(Number_T& new_left, Number_T& new_right, const Number_T& left, const Number_T& right) {
    verbose("coerce");
    // Return a Number_T 
    if (left.t == right.t) {
      new_left = left;
      new_right = right;
      return;
    }
    PyObject* o;
    switch(left.t) {
    case LONG:
      switch(right.t) {
      case DOUBLE:
	new_left = (double)left;
	new_right = right;
	return;
      case PYOBJECT:
	verbose("About to coerce a C long to an Integer");
	if (!(o = PyObject_CallFunction(pyfunc_Integer, "l", left.v._long))) {
	  py_error("Error coercing a long to an Integer");
	}
	new_left = o;
	//Py_DECREF(o);
	//new_left = PyInt_FromLong(left);
	new_right = right;
	return;
      default:
	std::cerr << "type = " << right.t << "\n";
	stub("** invalid coercion -- left LONG**");
      }
    case DOUBLE:
      switch(right.t) {
      case LONG:
	new_left = left;
	new_right = (double) right;
	return;
      case PYOBJECT:
	new_left = PyFloat_FromDouble(left);
	new_right = right;
	return;
      default:
	std::cerr << "type = " << right.t << "\n";
	stub("** invalid coercion -- left DOUBLE ** ");
      }
    case PYOBJECT:
      new_right = Number_T_to_pyobject(right);
      return;
    }
    std::cerr << "type = " << left.t << "\n";
    stub("** invalid coercion **");
  }

} // namespace GiNaC
