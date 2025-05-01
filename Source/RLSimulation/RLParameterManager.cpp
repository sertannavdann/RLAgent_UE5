// RLParameterManager.cpp
#include "RLParameterManager.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

URLParameterManager::URLParameterManager()
{
    PrimaryComponentTick.bCanEverTick = false;
    CurrentLearningRate = InitialLearningRate;
}

void URLParameterManager::AdjustLearningRate(float ErrorDelta)
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Adjust Learning Rate");
    
    if (!UseAdaptiveLearningRate)
        return;
    
    // Check if learning has stabilized
    if (HasLearningStabilized(ErrorDelta))
    {
        // Reduce learning rate
        float OldValue = CurrentLearningRate;
        CurrentLearningRate = FMath::Max(MinLearningRate, CurrentLearningRate / LearningRateReductionFactor);
        
        // Reset stability counter
        StableLearningSteps = 0;
        
        // Broadcast event
        OnLearningRateChanged.Broadcast(OldValue, CurrentLearningRate);
    }
    
    PreviousError = ErrorDelta;
}

bool URLParameterManager::HasLearningStabilized(float ErrorDelta)
{
    // Simple stabilization check: error is small and hasn't changed much
    const float ErrorThreshold = 0.01f;
    const float ErrorDeltaThreshold = 0.005f;
    const int32 StableStepsRequired = 100;
    
    bool IsStable = FMath::Abs(ErrorDelta) < ErrorThreshold && 
                   FMath::Abs(ErrorDelta - PreviousError) < ErrorDeltaThreshold;
    
    if (IsStable)
    {
        StableLearningSteps++;
        return StableLearningSteps >= StableStepsRequired;
    }
    else
    {
        StableLearningSteps = 0;
        return false;
    }
}

void URLParameterManager::ResetEligibilityTraces()
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Reset Eligibility Traces");
    EligibilityTraces.Empty();
}

void URLParameterManager::UpdateEligibilityTraces(const TArray<FString>& StateKeys, bool Replacing)
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Update Eligibility Traces");
    
    // Decay all existing traces
    for (auto& Pair : EligibilityTraces)
    {
        Pair.Value *= EligibilityDecayRate;
    }
    
    // Update traces for visited states
    for (const FString& StateKey : StateKeys)
    {
        if (Replacing)
        {
            // Replacing traces: set to 1
            EligibilityTraces.Add(StateKey, 1.0f);
        }
        else
        {
            // Accumulating traces: increment by 1
            float CurrentValue = 0.0f;
            if (EligibilityTraces.Contains(StateKey))
            {
                CurrentValue = EligibilityTraces[StateKey];
            }
            EligibilityTraces.Add(StateKey, CurrentValue + 1.0f);
        }
    }
    
    // Remove traces below threshold for efficiency
    for (auto It = EligibilityTraces.CreateIterator(); It; ++It)
    {
        if (FMath::Abs(It.Value()) < EligibilityThreshold)
        {
            It.RemoveCurrent();
        }
    }
}

float URLParameterManager::GetEligibilityValue(const FString& StateKey)
{
    if (EligibilityTraces.Contains(StateKey))
    {
        return EligibilityTraces[StateKey];
    }
    return 0.0f;
}

void URLParameterManager::GetTopEligibilityTraces(TArray<FString>& OutKeys, TArray<float>& OutValues, int32 MaxCount) const
{
    // Clear output arrays
    OutKeys.Empty();
    OutValues.Empty();
    
    // Create a temporary array of pairs for sorting
    TArray<TPair<FString, float>> SortedTraces;
    for (const auto& Pair : EligibilityTraces)
    {
        SortedTraces.Add(TPair<FString, float>(Pair.Key, Pair.Value));
    }
    
    // Sort by value (descending)
    SortedTraces.Sort([](const TPair<FString, float>& A, const TPair<FString, float>& B) {
        return A.Value > B.Value;
    });
    
    // Take top N
    int32 Count = FMath::Min(MaxCount, SortedTraces.Num());
    for (int32 i = 0; i < Count; i++)
    {
        OutKeys.Add(SortedTraces[i].Key);
        OutValues.Add(SortedTraces[i].Value);
    }
}