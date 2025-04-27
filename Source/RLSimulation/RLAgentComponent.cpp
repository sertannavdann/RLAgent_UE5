#include "RLAgentComponent.h"
#include "RLAgentManager.h"
#include "GameFramework/Actor.h"

URLAgentComponent::URLAgentComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void URLAgentComponent::BeginPlay()
{
    Super::BeginPlay();
}

void URLAgentComponent::InitializeAgent(ARLAgentManager* Manager)
{
    AgentManager = Manager;
    AgentID = AgentManager->RegisterAgent(this);

    // Simplified feature dimensionality: position (x,y) and direction (x,y)
    const int32 NumFeatures = 4;
    CurrentState.Features.Init(0.f, NumFeatures);
    ValueWeights.Init(0.f, NumFeatures);
    
    // Remove eligibility traces for simplicity
    
    // Record starting location
    PreviousLocation = GetOwner()->GetActorLocation();
}
