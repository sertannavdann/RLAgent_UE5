#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RLAgentManager.generated.h"

class URLAgentComponent;

USTRUCT(BlueprintType)
struct FAgentState
{
    GENERATED_BODY()
    
    UPROPERTY(BlueprintReadWrite)
    TArray<float> Features;
};


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
    
    /** TD learning parameters */
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

protected:
    virtual void Tick(float DeltaTime) override;

private:
    /** All registered agents */
    UPROPERTY()
    TArray<URLAgentComponent*> Agents;

    /** Current exploration rate */
    float CurrentEpsilon = 1.f;

    /** Dimensionality of state features */
    int32 NumStateFeatures = 6;

    /** Sample the environment to produce a feature vector */
    TArray<float> GetStateFeatures(URLAgentComponent* Agent);

    /** Execute agent Action and return new state */
    TArray<float> GetNextState(URLAgentComponent* Agent, int32 Action);

    /** Reward function based on movement toward world origin */
    float CalculateReward(URLAgentComponent* Agent, const TArray<float>& NextState);

    /** Linear value estimation v(s)=w·x */
    float ComputeValue(const TArray<float>& Weights, const TArray<float>& State);

    /** One TD(λ) update */
    void TDUpdate(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState);
    
    TArray<float> GetPotentialState(URLAgentComponent* Agent, int32 Action);

};
