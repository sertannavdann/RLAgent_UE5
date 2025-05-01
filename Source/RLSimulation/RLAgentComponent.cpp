#include "RLAgentComponent.h"
#include "RLAgentManager.h"
#include "RLParameterManager.h"
#include "GameFramework/Actor.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

URLAgentComponent::URLAgentComponent(): PreviousLocation(), ParameterManager(nullptr)
{
    PrimaryComponentTick.bCanEverTick = false;

    // Initialize history array for monitoring
    RewardHistory.Init(0.0f, 100);
}

void URLAgentComponent::BeginPlay()
{
    Super::BeginPlay();
    
    // Try to find ParameterManager on the owner actor
    if (!ParameterManager)
    {
        ParameterManager = Cast<URLParameterManager>(
            GetOwner()->FindComponentByClass(URLParameterManager::StaticClass()));
        
        // If not found, create one
        if (!ParameterManager)
        {
            ParameterManager = NewObject<URLParameterManager>(GetOwner());
            ParameterManager->RegisterComponent();
        }
    }
}

void URLAgentComponent::InitializeAgent(ARLAgentManager* Manager)
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Initialize Agent");
    
    AgentManager = Manager;
    AgentID = AgentManager->RegisterAgent(this);

    // Full 3D feature dimensionality: position (x,y,z) and direction (x,y,z)
    constexpr int32 NumFeatures = 6;
    CurrentState.Features.Init(0.f, NumFeatures);
    ValueWeights.Init(0.f, NumFeatures);
    EligibilityTraces.Init(0.f, NumFeatures);
    
    // Record starting location
    PreviousLocation = GetOwner()->GetActorLocation();
    
    // Initialize parameter manager if needed
    if (!ParameterManager)
    {
        BeginPlay(); // Will handle parameter manager initialization
    }
    
    // Reset stats
    StepsTaken = 0;
    TargetsFound = 0;
    RewardHistory.Init(0.0f, 100);
}