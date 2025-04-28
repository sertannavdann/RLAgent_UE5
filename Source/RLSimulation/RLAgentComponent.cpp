#include "RLAgentComponent.h"
#include "RLAgentManager.h"
#include "GameFramework/Actor.h"

URLAgentComponent::URLAgentComponent(): PreviousLocation()
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

    // Full 3D feature dimensionality: position (x,y,z) and direction (x,y,z)
    constexpr int32 NumFeatures = 6;
    CurrentState.Features.Init(0.f, NumFeatures);
    ValueWeights.Init(0.f, NumFeatures);
    
    // Record starting location
    PreviousLocation = GetOwner()->GetActorLocation();
}