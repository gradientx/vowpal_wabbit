#include <jni.h>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include "../vowpalwabbit/vw.h"
#include "../vowpalwabbit/constant.h"

#include "com_gradientx_vw_JniVowpalWabbit.h"
using namespace std;
 
/* global VW instance */
#define VW_NUM com_gradientx_vw_JniVowpalWabbit_VW_NUM
vw* vw_arr[VW_NUM];

JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_initialize(JNIEnv *env, jobject, jint id, jstring initStr)
{
  char *str;
    str = (char*) (env)->GetStringUTFChars(initStr, NULL);
  if (str == NULL) {
    return; /* OutOfMemoryError already thrown */
  }
  vw_arr[id] = VW::initialize(str);
  (env)->ReleaseStringUTFChars(initStr, str);
}

JNIEXPORT jfloat JNICALL Java_com_gradientx_vw_JniVowpalWabbit_read_1example(JNIEnv *env, jobject, jint id, jstring exStr)
{
  char *str;
  str = (char*) (env)->GetStringUTFChars(exStr, NULL);
  if (str == NULL) {
    return -1.0f; /* OutOfMemoryError already thrown */
  }
    
  vw* model = vw_arr[id];
  example *vec = VW::read_example(*model, str);
  model->learn(vec);
  float ret = vec->final_prediction;
  VW::finish_example(*model, vec);

  (env)->ReleaseStringUTFChars(exStr, str);
  return ret;
}

JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_finish(JNIEnv *, jobject, jint id)
{
  vw* model = vw_arr[id];
  VW::finish(*model);
}

JNIEXPORT jfloatArray JNICALL Java_com_gradientx_vw_JniVowpalWabbit_get_1coef(JNIEnv *env, jobject, jint id, jint size, jint constIdx)
{
  jfloatArray result;
  result = (env)->NewFloatArray(size);
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }

  vw* model = vw_arr[id];

  uint32_t length = 1 << model->num_bits;
  size_t stride = model->reg.stride;

  jfloat arr[size];
  for (int i=0; i<size; i++) arr[i] = model->reg.weight_vector[stride*i];
  if (constIdx>=0) {
    int ci = ((constant*stride)&model->reg.weight_mask)/stride;
    arr[constIdx]=model->reg.weight_vector[stride*ci];
  }
    
  (env)->SetFloatArrayRegion(result,0,size,arr);
  return(result);
}

void dump_regressor(vw& all, string reg_name, bool as_text, bool reg_vector);
void read_vector(vw& all, const char* file, bool& initialized, bool reg_vector);

JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_dump_1regressor
(JNIEnv *env, jobject, jint id, jstring fn)
{
  char *str;
  str = (char*) (env)->GetStringUTFChars(fn, NULL);
  if (str == NULL) {
    return; /* OutOfMemoryError already thrown */
  }
    
  vw* model = vw_arr[id];
  dump_regressor(*model,string(str),false,false);

  (env)->ReleaseStringUTFChars(fn, str);
}

JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_load_1regressor
(JNIEnv *env, jobject, jint id, jstring fn)
{
  char *str;
  str = (char*) (env)->GetStringUTFChars(fn, NULL);
  if (str == NULL) {
    return; /* OutOfMemoryError already thrown */
  }

  bool initialized = false;
  vw* model = vw_arr[id];
  read_vector(*model, str, initialized, false);

  (env)->ReleaseStringUTFChars(fn, str);
}

JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_set_1coef
(JNIEnv *env, jobject, jint id, jfloatArray coef, jint constIdx)
{
  vw* model = vw_arr[id];

  uint32_t length = 1 << model->num_bits;
  size_t stride = model->reg.stride;

  jfloat *body = (env)->GetFloatArrayElements(coef, 0);
  int size=(env)->GetArrayLength(coef);
  for (int i=0; i<size; i++) if (i!=constIdx) model->reg.weight_vector[stride*i] = body[i];
  if (constIdx>=0) {
    int ci = ((constant*stride)&model->reg.weight_mask)/stride;
    model->reg.weight_vector[stride*ci] = body[constIdx];
  }

  (env)->ReleaseFloatArrayElements(coef, body, 0);

}

