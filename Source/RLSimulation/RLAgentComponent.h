#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RLAgentManager.h"
#include "GameFramework/SaveGame.h"
#include "RLAgentComponent.generated.h"

class ARLAgentManager;

// Save game object to cache trained agents
UCLASS()
class RLSIMULATION_API URLAgentSaveGame : public USaveGame
{
    GENERATED_BODY()
    
public:
    UPROPERTY(BlueprintReadWrite, Category = "Reinforcement Learning")
    TArray<float> ValueWeights;
    
    UPROPERTY(BlueprintReadWrite, Category = "Reinforcement Learning")
    float TotalTrainingTime;
    
    UPROPERTY(BlueprintReadWrite, Category = "Reinforcement Learning")
    int32 TrainingEpisodes;
    
    UPROPERTY(BlueprintReadWrite, Category = "Reinforcement Learning")
    FString AgentDescription;
};

UCLASS(ClassGroup=RL, meta=(BlueprintSpawnableComponent))
class RLSIMULATION_API URLAgentComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    URLAgentComponent();

    /** Called by the manager to register this agent */
    UFUNCTION(BlueprintCallable, Category="RL")
    void InitializeAgent(ARLAgentManager* Manager);

    /** Current observed state */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    FAgentState CurrentState;

    /** Linear value-function weights */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    TArray<float> ValueWeights;

    /** Eligibility traces for TD(λ) */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    TArray<float> EligibilityTraces;

    /** Cumulative reward in current episode */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    float CumulativeReward = 0.f;

    /** Agent's previous world location (for reward calculation) */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    FVector PreviousLocation;

    /** Assigned by manager */
    UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="RL")
    int32 AgentID = -1;

protected:
    virtual void BeginPlay() override;

private:
    UPROPERTY()
    ARLAgentManager* AgentManager = nullptr;
};
