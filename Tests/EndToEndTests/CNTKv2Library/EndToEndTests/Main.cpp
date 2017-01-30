//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

#include <iostream>
#include <vector>

using namespace CNTK;

void TrainCifarResnet();
void TrainLSTMSequenceClassifer();
void MNISTClassifierTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifer();

int main(int argc, char *argv[])
{
    if (argc > 2)
    {
        fprintf(stderr, "Wrong number of arguments.\n");
        return -1;
    }

#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

#ifndef CPUONLY
    fprintf(stderr, "Run tests using GPU build.\n");
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    std::string testName(argv[1]);

    if (!testName.compare("CifarResNet"))
    {
        if (IsGPUAvailable())
        {
            fprintf(stderr, "Run test on a GPU device.\n");
            TrainCifarResnet();
        }
        else
        {
            fprintf(stderr, "Cannot run TrainCifarResnet test on a CPU device.\n");
        }
    }
    else if (!testName.compare("LSTMSequenceClassifier"))
    {
        TrainLSTMSequenceClassifer();
    }
    else if (!testName.compare("MNISTClassifier"))
    {
        MNISTClassifierTests();
    }
    else if (!testName.compare("SequenceToSequence"))
    {
        TrainSequenceToSequenceTranslator();
    }
    else if (!testName.compare("TruncatedLSTMAcousticModel"))
    {
        TrainTruncatedLSTMAcousticModelClassifer();
    }
    else
    {
        fprintf(stderr, "End to end test not found.\n");
        return -1;
    }

    std::string testsPassedMsg = "\nCNTKv2Library-" + testName + " tests: Passed\n";

    fprintf(stderr, "%s", testsPassedMsg.c_str());
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif
}
