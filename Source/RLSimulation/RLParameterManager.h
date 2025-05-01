
#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RLParameterManager.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnLearningRateChanged, float, OldValue, float, NewValue);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnEligibilityDecayChanged, float, NewValue);

UCLASS(ClassGroup=RL, meta=(BlueprintSpawnableComponent))
class RLSIMULATION_API URLParameterManager : public UActorComponent
{
    GENERATED_BODY()

public:
    URLParameterManager();
    
    // Learning rate properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Learning Rate")
    float InitialLearningRate = 0.1f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Learning Rate")
    float MinLearningRate = 0.00001f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Learning Rate")
    float LearningRateReductionFactor = 2.0f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Learning Rate")
    bool UseAdaptiveLearningRate = true;
    
    UPROPERTY(BlueprintReadOnly, Category="RL|Learning Rate")
    float CurrentLearningRate;
    
    // Eligibility trace properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Eligibility Traces")
    float EligibilityDecayRate = 0.9f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Eligibility Traces")
    float EligibilityThreshold = 0.01f;
    
    // Events
    UPROPERTY(BlueprintAssignable, Category="RL|Events")
    FOnLearningRateChanged OnLearningRateChanged;
    
    UPROPERTY(BlueprintAssignable, Category="RL|Events")
    FOnEligibilityDecayChanged OnEligibilityDecayChanged;
    
    // Function to adjust learning rate
    UFUNCTION(BlueprintCallable, Category="RL|Learning Rate")
    void AdjustLearningRate(float ErrorDelta);
    
    // Reset traces for new episode
    UFUNCTION(BlueprintCallable, Category="RL|Eligibility Traces")
    void ResetEligibilityTraces();
    
    // Update eligibility traces
    UFUNCTION(BlueprintCallable, Category="RL|Eligibility Traces")
    void UpdateEligibilityTraces(const TArray<FString>& StateKeys, bool Replacing = false);
    
    // Get eligibility value for a state
    UFUNCTION(BlueprintCallable, Category="RL|Eligibility Traces")
    float GetEligibilityValue(const FString& StateKey);

    // Get current learning rate
    UFUNCTION(BlueprintPure, Category="RL|Learning Rate")
    float GetCurrentLearningRate() const { return CurrentLearningRate; }

    // Get all eligibility traces for visualization
    UFUNCTION(BlueprintCallable, Category="RL|Debugging")
    void GetTopEligibilityTraces(TArray<FString>& OutKeys, TArray<float>& OutValues, int32 MaxCount = 5) const;

private:
    UPROPERTY()
    TMap<FString, float> EligibilityTraces;
    
    float PreviousError = 0.0f;
    int32 StableLearningSteps = 0;
    
    // Diagnostic test for learning stability
    bool HasLearningStabilized(float ErrorDelta);
};