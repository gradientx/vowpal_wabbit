/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_gradientx_vw_JniVowpalWabbit */

#ifndef _Included_com_gradientx_vw_JniVowpalWabbit
#define _Included_com_gradientx_vw_JniVowpalWabbit
#ifdef __cplusplus
extern "C" {
#endif
#undef com_gradientx_vw_JniVowpalWabbit_VW_NUM
#define com_gradientx_vw_JniVowpalWabbit_VW_NUM 1000L
/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    initialize
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_initialize
  (JNIEnv *, jobject, jint, jstring);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    read_example
 * Signature: (ILjava/lang/String;)F
 */
JNIEXPORT jfloat JNICALL Java_com_gradientx_vw_JniVowpalWabbit_read_1example
  (JNIEnv *, jobject, jint, jstring);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    finish
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_finish
  (JNIEnv *, jobject, jint);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    get_coef
 * Signature: (III)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_gradientx_vw_JniVowpalWabbit_get_1coef
  (JNIEnv *, jobject, jint, jint, jint);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    dump_regressor
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_dump_1regressor
  (JNIEnv *, jobject, jint, jstring);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    load_regressor
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_load_1regressor
  (JNIEnv *, jobject, jint, jstring);

/*
 * Class:     com_gradientx_vw_JniVowpalWabbit
 * Method:    set_coef
 * Signature: (I[FI)V
 */
JNIEXPORT void JNICALL Java_com_gradientx_vw_JniVowpalWabbit_set_1coef
  (JNIEnv *, jobject, jint, jfloatArray, jint);

#ifdef __cplusplus
}
#endif
#endif
