import os,textwrap as T
B="/home/bbrelin/nimcp"
def w(p,c):
 f=os.path.join(B,p);os.makedirs(os.path.dirname(f),exist_ok=1);open(f,"w").write(T.dedent(c).lstrip());print("OK:"+p)
w("test/unit/security/test_bbb_api_validation.cpp",r"""
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
extern "C" { #include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h" }
class BBBApiValidationTest : public ::testing::Test { protected:
    bbb_system_t system_ = nullptr;
    void SetUp() override { bbb_config_t c=bbb_default_config(); c.strict_mode=true;
      c.input.validate_strings=true; c.input.max_string_length=4096;
      system_=bbb_system_create(&c); bbb_system_set_enabled(system_,true); }
    void TearDown() override { if(system_){bbb_system_destroy(system_);system_=nullptr;} } };
TEST_F(BBBApiValidationTest, ValidInput_Success) { const char* d="hello";
  bbb_validation_result_t r; memset(&r,0,sizeof(r));
  EXPECT_TRUE(bbb_validate_input(system_,d,strlen(d),&r)); EXPECT_TRUE(r.valid); }
TEST_F(BBBApiValidationTest, NullData_Rejected) { bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_FALSE(bbb_validate_input(system_,nullptr,100,&r)); }
TEST_F(BBBApiValidationTest, NullSystem_Rejected) { bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_FALSE(bbb_validate_input(nullptr,"x",1,&r)); }
TEST_F(BBBApiValidationTest, OversizedBuffer) { std::vector<char> b(8192,65);
  bbb_validation_result_t r; memset(&r,0,sizeof(r));
  bbb_validate_input(system_,b.data(),b.size(),&r); }
TEST_F(BBBApiValidationTest, ValidString_OK) { bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_TRUE(bbb_validate_string(system_,"safe",&r)); }
TEST_F(BBBApiValidationTest, NullString_Rejected) { bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_FALSE(bbb_validate_string(system_,nullptr,&r)); }
TEST_F(BBBApiValidationTest, OversizedString_Rejected) { std::string s(8192,88);
  bbb_validation_result_t r; memset(&r,0,sizeof(r));
  EXPECT_FALSE(bbb_validate_string(system_,s.c_str(),&r)); }
TEST_F(BBBApiValidationTest, ValidPointer_OK) { int v=42; bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_TRUE(bbb_validate_pointer(system_,&v,sizeof(int),&r)); }
TEST_F(BBBApiValidationTest, NullPointer_Rejected) { bbb_validation_result_t r;
  memset(&r,0,sizeof(r)); EXPECT_FALSE(bbb_validate_pointer(system_,nullptr,4,&r)); }
class BBBHelpersTest : public ::testing::Test { protected:
  void SetUp() override { bbb_helpers_init(); }
  void TearDown() override { bbb_helpers_shutdown(); } };
TEST_F(BBBHelpersTest, RegisterModule_OK) { EXPECT_TRUE(bbb_register_module("test",BBB_MODULE_TYPE_CORE)); }
TEST_F(BBBHelpersTest, RegisterModule_Null) { EXPECT_FALSE(bbb_register_module(nullptr,BBB_MODULE_TYPE_CORE)); }
TEST_F(BBBHelpersTest, CheckPointer_Valid) { int v=1; EXPECT_TRUE(bbb_check_pointer(&v,"t")); }
TEST_F(BBBHelpersTest, CheckPointer_Null) { EXPECT_FALSE(bbb_check_pointer(nullptr,"t")); }
TEST_F(BBBHelpersTest, CheckString_Valid) { EXPECT_TRUE(bbb_check_string("hi",256,"t")); }
TEST_F(BBBHelpersTest, CheckString_Null) { EXPECT_FALSE(bbb_check_string(nullptr,256,"t")); }
TEST_F(BBBHelpersTest, ValidateRange_OK) { EXPECT_TRUE(bbb_validate_range(50,0,100,"t")); }
TEST_F(BBBHelpersTest, ValidateRange_Bad) { EXPECT_FALSE(bbb_validate_range(200,0,100,"t")); }
TEST(BBBSystem, Create_Default) { bbb_system_t s=bbb_system_create(nullptr); EXPECT_NE(s,nullptr); if(s)bbb_system_destroy(s); }
TEST(BBBSystem, Destroy_Null) { bbb_system_destroy(nullptr); }
TEST(BBBSystem, Stats_Null) { bbb_statistics_t s; EXPECT_FALSE(bbb_system_get_statistics(nullptr,&s)); }
""")
w("test/unit/security/test_security_error_codes.cpp",r"""
#include <gtest/gtest.h>
#include <cstring>
extern "C" { #include "security/nimcp_security.h"
#include "security/nimcp_post_quantum.h"
#include "security/nimcp_security_consensus.h"
#include "utils/error/nimcp_error_codes.h" }
#define EXPECT_VALID_NIMCP(e) do{auto _r=(nimcp_error_t)(e);EXPECT_NE(_r,(nimcp_error_t)-1)<<#e " returned -1";if(_r\!=0)EXPECT_GE(_r,1);}while(0)
TEST(SecErrCode, DirectiveAdd_Null) { EXPECT_VALID_NIMCP(nimcp_directive_add(nullptr,"test")); }
TEST(SecErrCode, DirectiveLock_Null) { EXPECT_VALID_NIMCP(nimcp_directive_lock(nullptr)); }
TEST(SecErrCode, EncryptNull) { uint8_t d[]={1};uint8_t o[256];size_t s=0;
  EXPECT_VALID_NIMCP(nimcp_encryption_encrypt(nullptr,d,1,o,256,&s)); }
TEST(SecErrCode, DecryptNull) { uint8_t d[]={1};uint8_t o[256];size_t s=0;
  EXPECT_VALID_NIMCP(nimcp_encryption_decrypt(nullptr,d,1,o,256,&s)); }
TEST(SecErrCode, GenKeyNull) { EXPECT_VALID_NIMCP(nimcp_encryption_generate_key(nullptr)); }
