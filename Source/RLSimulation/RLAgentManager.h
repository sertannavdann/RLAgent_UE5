#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RLAgentManager.generated.h"

class URLAgentComponent;
class URLParameterManager;

UCLASS()
class RLSIMULATION_API ARLAgentManager : public AActor
{
    GENERATED_BODY()

public:
    ARLAgentManager();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Environment|TargetPosition")
    FVector TargetObjectPosition;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Environment|TestArea")
    float SphereRadius;
    
    UFUNCTION(BlueprintCallable, Category="Environment")
    void RespawnTarget();
    
public:
    /** Called by agent components to register themselves */
    UFUNCTION(BlueprintCallable, Category="RL")
    int32 RegisterAgent(URLAgentComponent* Agent);

    /** Begin the training process */
    UFUNCTION(BlueprintCallable, Category="RL")
    void StartTraining();

    /** How many floats compose the state feature vector */
    UFUNCTION(BlueprintCallable, Category="RL")
    int32 GetNumStateFeatures() const { return NumStateFeatures; }
    
    /** TD learning parameters - kept for backward compatibility */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL")
    float LearningRate = 0.1f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL")
    float DiscountFactor = 0.9f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL")
    float Lambda = 0.8f;

    /** ε-greedy parameters */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL")
    float InitialEpsilon = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL")
    float EpsilonDecay = 0.995f;
    
    /** Enable/disable eligibility traces */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Advanced")
    bool bUseEligibilityTraces = true;
    
    /** Type of eligibility traces */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Advanced")
    bool bUseReplacingTraces = false;
    
    /** Parameter manager for global settings */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    URLParameterManager* GlobalParameterManager;
    
    /** Calculate TD error for an agent (for debugging) */
    UFUNCTION(BlueprintCallable, Category="RL|Debugging")
    float CalculateTDError(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState) const;

protected:
    virtual void Tick(float DeltaTime) override;
    virtual void BeginPlay() override;

private:
    /** All registered agents */
    UPROPERTY()
    TArray<URLAgentComponent*> Agents;

    /** Current exploration rate */
    float CurrentEpsilon = 1.f;

    /** Dimensionality of state features */
    int32 NumStateFeatures = 6; // Updated to 6 for full 3D

    /** Sample the environment to produce a feature vector */
    static TArray<float> GetStateFeatures(const URLAgentComponent* Agent);

    /** Execute agent Action and return new state */
    static TArray<float> GetNextState(const URLAgentComponent* Agent, int32 Action);

    /** Reward function based on movement toward target */
    float CalculateReward(const URLAgentComponent* Agent, const TArray<float>& NextState) const;

    /** Linear value estimation v(s)=w·x */
    static float ComputeValue(const TArray<float>& Weights, const TArray<float>& State);

    /** One TD update with eligibility traces */
    void TDUpdate(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState) const;
    
    static TArray<float> GetPotentialState(const URLAgentComponent* Agent, int32 Action);

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Environment")
    float SphereRadius_InEditor = 500.f;

    /** Returns the global sphere radius. */
    UFUNCTION(BlueprintPure, Category="RL|Environment")
    static float GetSphereRadius();

    /** Sets the global sphere radius. */
    UFUNCTION(BlueprintCallable, Category="RL|Environment")
    static void SetSphereRadius(float NewRadius);

private:
    static float G_SphereRadius; // "real" storage of the radius
    
    // Performance monitoring for MacOS
    double FrameStartTime = 0.0;
    double TDUpdateTime = 0.0;
    int32 UpdateCount = 0;
};