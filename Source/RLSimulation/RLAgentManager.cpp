#include "RLAgentManager.h"
#include "RLAgentComponent.h"
#include "RLParameterManager.h"
#include "DrawDebugHelpers.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "Kismet/KismetSystemLibrary.h"

float ARLAgentManager::G_SphereRadius = 500.f;

ARLAgentManager::ARLAgentManager()
{
    PrimaryActorTick.bCanEverTick = true;
    
    // Set up the target object position in full 3D space
    TargetObjectPosition = FVector(FMath::RandRange(-1000.f, 1000.f),
                                  FMath::RandRange(-1000.f, 1000.f),
                                  FMath::RandRange(-1000.f, 1000.f));
    
    // Sphere radius represents the training area
    SphereRadius = 1500.f;
    
    // Simplified learning parameters
    InitialEpsilon = 0.3f;    // Less exploration
    EpsilonDecay = 0.999f;    // Slower decay
    LearningRate = 0.1f;      // Simple learning rate
    DiscountFactor = 0.9f;    // Standard discount
    
    // Create parameter manager
    GlobalParameterManager = CreateDefaultSubobject<URLParameterManager>(TEXT("ParameterManager"));
    
    // Update NumStateFeatures to 6 for full 3D
    NumStateFeatures = 6;
}

void ARLAgentManager::BeginPlay()
{
    Super::BeginPlay();
    
    // Initialize the global parameter manager if needed
    if (!GlobalParameterManager)
    {
        GlobalParameterManager = NewObject<URLParameterManager>(this);
        GlobalParameterManager->RegisterComponent();
    }
    
    // Set initial learning rate from the global parameter manager
    LearningRate = GlobalParameterManager->InitialLearningRate;
}

void ARLAgentManager::RespawnTarget()
{
    // Clear old target debug markers
    FlushPersistentDebugLines(GetWorld());
    
    // Generate new random position within sphere using 3D spherical coordinates
    float Radius = FMath::RandRange(0.f, SphereRadius * 0.8f);
    float Theta = FMath::RandRange(0.f, 2.f * PI); // Azimuthal angle
    float Phi = FMath::RandRange(0.f, PI);        // Polar angle
    
    TargetObjectPosition = FVector(
        Radius * FMath::Sin(Phi) * FMath::Cos(Theta),
        Radius * FMath::Sin(Phi) * FMath::Sin(Theta),
        Radius * FMath::Cos(Phi)
    );
    
    // Draw the new target
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, true);
}

int32 ARLAgentManager::RegisterAgent(URLAgentComponent* Agent)
{
    Agents.Add(Agent);
    return Agents.Num() - 1;
}

void ARLAgentManager::StartTraining()
{
    CurrentEpsilon = InitialEpsilon;
    
    // Reset eligibility traces for all agents
    for (URLAgentComponent* Agent : Agents)
    {
        if (Agent->ParameterManager)
        {
            Agent->ParameterManager->ResetEligibilityTraces();
        }
    }
    
    // Spawn a debug marker at the target position
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, true);
    
    // Initialize performance monitoring
    FrameStartTime = FPlatformTime::Seconds();
    TDUpdateTime = 0.0;
    UpdateCount = 0;
}

void ARLAgentManager::Tick(float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("RL Agent Manager Tick");
    
    Super::Tick(DeltaTime);

    if (Agents.Num() == 0) return;
    
    // Start measuring frame time
    FrameStartTime = FPlatformTime::Seconds();

    for (URLAgentComponent* Agent : Agents)
    {
        // 1. Observe current state
        TArray<float> CurrentFeatures = GetStateFeatures(Agent);
        Agent->CurrentState.Features = CurrentFeatures;

        // 2. Choose action: Expanded to 5 options (forward, left, right, up, down)
        int32 Action = 0;
        if (FMath::FRand() < CurrentEpsilon)
        {
            // Random exploration - pick any action (now including up/down)
            Action = FMath::RandRange(0, 4);
        }
        else
        {
            // Action selection - evaluate each action
            float BestValue = -FLT_MAX;
            for (int32 PotentialAction = 0; PotentialAction < 5; PotentialAction++)
            {
                const TArray<float> PotentialState = GetPotentialState(Agent, PotentialAction);
                const float Value = ComputeValue(Agent->ValueWeights, PotentialState);
                if (Value > BestValue)
                {
                    BestValue = Value;
                    Action = PotentialAction;
                }
            }
        }
        
        // 3. Execute action and get next state
        TArray<float> NextFeatures = GetNextState(Agent, Action);
        
        // 4. Calculate reward based on finding the target object
        float Reward = CalculateReward(Agent, NextFeatures);

        // 5. TD Update with eligibility traces
        double UpdateStartTime = FPlatformTime::Seconds();
        TDUpdate(Agent, Reward, NextFeatures);
        TDUpdateTime += (FPlatformTime::Seconds() - UpdateStartTime);
        UpdateCount++;

        // 6. Update state and bookkeeping
        Agent->CurrentState.Features = NextFeatures;
        Agent->CumulativeReward += Reward;
        Agent->PreviousLocation = Agent->GetOwner()->GetActorLocation();
        Agent->StepsTaken++;
        
        // Update reward history for monitoring
        if (Agent->RewardHistory.Num() > 0)
        {
            Agent->RewardHistory.RemoveAt(0);
            Agent->RewardHistory.Add(Reward);
        }

        // 7. Draw value above agent for debugging
        float Val = ComputeValue(Agent->ValueWeights, NextFeatures);
        DrawDebugString(GetWorld(),
                        Agent->GetOwner()->GetActorLocation() + FVector(0,0,100),
                        FString::Printf(TEXT("V=%.2f"), Val),
                        nullptr, FColor::White, 0.f, true);
                        
        // 8. Check if agent found the target
        FVector CurrentLocation = Agent->GetOwner()->GetActorLocation();
        if (FVector::Dist(CurrentLocation, TargetObjectPosition) < 100.f)
        {
            // Agent found the target! Give big reward and respawn target
            Agent->CumulativeReward += 50.f;
            Agent->TargetsFound++;
            RespawnTarget();
            
            // Reset eligibility traces for new episode
            if (Agent->ParameterManager)
            {
                Agent->ParameterManager->ResetEligibilityTraces();
            }
        }
    }

    // Reduce exploration over time (more slowly)
    CurrentEpsilon = FMath::Max(0.05f, CurrentEpsilon * EpsilonDecay);
    
    // Draw the training area - 3D sphere
    DrawDebugSphere(GetWorld(), FVector::ZeroVector, SphereRadius, 24, FColor::Green, false, -1, 0, 2.0f);
    
    // Draw the target object
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, false, -1, 0, 5.0f);
    
    // Performance monitoring on screen
    if (UpdateCount > 0 && (UpdateCount % 100 == 0))
    {
        float AvgUpdateTime = TDUpdateTime / UpdateCount * 1000.0f; // Convert to ms
        FString DebugText = FString::Printf(TEXT("Avg TD Update: %.3f ms | Epsilon: %.3f"), 
                                           AvgUpdateTime, CurrentEpsilon);
        UKismetSystemLibrary::PrintString(GetWorld(), DebugText, true, false, FLinearColor::Yellow, 2.0f);
        
        // Reset counters periodically
        if (UpdateCount >= 1000)
        {
            TDUpdateTime = 0.0;
            UpdateCount = 0;
        }
    }
}

TArray<float> ARLAgentManager::GetStateFeatures(const URLAgentComponent* Agent)
{
    // Full 3D state: position (x,y,z) and direction (x,y,z)
    FVector Loc = Agent->GetOwner()->GetActorLocation();
    FVector Forward = Agent->GetOwner()->GetActorForwardVector();

    return {
        static_cast<float>(Loc.X / 1000.f),       // Normalized X position
        static_cast<float>(Loc.Y / 1000.f),       // Normalized Y position
        static_cast<float>(Loc.Z / 1000.f),       // Normalized Z position
        static_cast<float>(Forward.X),            // Direction X
        static_cast<float>(Forward.Y),            // Direction Y
        static_cast<float>(Forward.Z)             // Direction Z
    };
}

TArray<float> ARLAgentManager::GetPotentialState(const URLAgentComponent* Agent, int32 Action)
{
    // Get current state without modifying the actor
    FVector CurrentLoc = Agent->GetOwner()->GetActorLocation();
    FRotator CurrentRot = Agent->GetOwner()->GetActorRotation();

    // Simulate action in 3D space
    switch(Action)
    {
        case 0: // Move forward
            CurrentLoc += CurrentRot.Vector() * 100.f;
            break;
        case 1: // Turn left
            CurrentRot.Yaw -= 15.f;
            break;
        case 2: // Turn right
            CurrentRot.Yaw += 15.f;
            break;
        case 3: // Move up
            CurrentLoc.Z += 100.f;
            break;
        case 4: // Move down
            CurrentLoc.Z -= 100.f;
            break;
        default:
            break;
    }

    // Return full 3D features [X, Y, Z, ForwardX, ForwardY, ForwardZ]
    FVector SimulatedForward = CurrentRot.Vector();
    return {
        static_cast<float>(CurrentLoc.X / 1000.f),
        static_cast<float>(CurrentLoc.Y / 1000.f),
        static_cast<float>(CurrentLoc.Z / 1000.f),
        static_cast<float>(SimulatedForward.X),
        static_cast<float>(SimulatedForward.Y),
        static_cast<float>(SimulatedForward.Z)
    };
}

float ARLAgentManager::GetSphereRadius()
{
    return G_SphereRadius;
}

void ARLAgentManager::SetSphereRadius(float NewRadius)
{
    G_SphereRadius = NewRadius;
}

TArray<float> ARLAgentManager::GetNextState(const URLAgentComponent* Agent, int32 Action)
{
    // Get the current sphere radius
    const float Radius = GetSphereRadius();

    FVector Loc = Agent->GetOwner()->GetActorLocation();

    // 3D Boundary check
    if (Loc.Size() > Radius)
    {
        FVector Dir = -Loc.GetSafeNormal();
        Loc = Dir * (Radius * 0.8f);
        Agent->GetOwner()->SetActorLocation(Loc);
    }

    // Apply action directly in 3D
    switch(Action)
    {
        case 0: // Move forward
            Loc += Agent->GetOwner()->GetActorForwardVector() * 100.f;
            break;
        case 1: // Turn left
            Agent->GetOwner()->AddActorLocalRotation(FRotator(0, -15, 0));
            break;
        case 2: // Turn right
            Agent->GetOwner()->AddActorLocalRotation(FRotator(0, 15, 0));
            break;
        case 3: // Move up
            Loc.Z += 100.f;
            break;
        case 4: // Move down
            Loc.Z -= 100.f;
            break;
        default:
            break;
    }
    
    Agent->GetOwner()->SetActorLocation(Loc);
    return GetStateFeatures(Agent);
}

float ARLAgentManager::CalculateReward(const URLAgentComponent* Agent, const TArray<float>& NextState) const
{
    // The main objective is to find the target object in 3D space
    const FVector CurrentLocation = Agent->GetOwner()->GetActorLocation();
    
    // Calculate distance to target
    float DistanceToTarget = FVector::Dist(CurrentLocation, TargetObjectPosition);
    
    // Calculate previous distance (to see if we're getting closer)
    float PreviousDistanceToTarget = FVector::Dist(Agent->PreviousLocation, TargetObjectPosition);
    
    // Reward for getting closer to the target
    float DistanceReward = (PreviousDistanceToTarget - DistanceToTarget) / 100.f;
    
    // Small penalty for being far from target (to encourage exploration toward target)
    float DistancePenalty = -DistanceToTarget / 5000.f;
    
    // Boundary penalty to keep agent inside sphere
    float BoundaryPenalty = (CurrentLocation.Size() > SphereRadius * 0.9f) ? -1.f : 0.f;
    
    // Big reward if very close to target (will be detected in Tick)
    float FoundTargetReward = (DistanceToTarget < 100.f) ? 10.f : 0.f;
    
    return DistanceReward + DistancePenalty + BoundaryPenalty + FoundTargetReward;
}

float ARLAgentManager::ComputeValue(const TArray<float>& Weights, const TArray<float>& State)
{
    // Simple linear function approximation
    float Sum = 0.f;
    for (int32 i = 0; i < FMath::Min(Weights.Num(), State.Num()); i++)
        Sum += Weights[i] * State[i];
    return Sum;
}

void ARLAgentManager::TDUpdate(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("TD Learning With Eligibility Traces");
    
    // Get current state value
    float V_current = ComputeValue(Agent->ValueWeights, Agent->CurrentState.Features);
    
    // Get next state value
    float V_next = ComputeValue(Agent->ValueWeights, NextState);
    
    // Calculate TD error
    float Delta = Reward + DiscountFactor * V_next - V_current;
    
    // Use parameters from the agent's parameter manager or fall back to global
    URLParameterManager* ParamManager = Agent->ParameterManager ? Agent->ParameterManager : GlobalParameterManager;
    
    if (ParamManager && bUseEligibilityTraces)
    {
        // Update eligibility traces
        TArray<FString> StateKeys;
        StateKeys.Add(FAgentState{Agent->CurrentState.Features}.GetKey());
        
        // Update traces (either accumulating or replacing)
        ParamManager->UpdateEligibilityTraces(StateKeys, bUseReplacingTraces);
        
        // Get the current learning rate from parameter manager
        float CurrentLearningRate = ParamManager->GetCurrentLearningRate();
        
        // Update all weights based on eligibility traces
        for (int32 i = 0; i < Agent->ValueWeights.Num(); i++)
        {
            // For simplicity, we directly update eligibility traces for the feature vector
            if (i < Agent->EligibilityTraces.Num())
            {
                if (bUseReplacingTraces)
                {
                    // Replacing traces
                    Agent->EligibilityTraces[i] = (i < Agent->CurrentState.Features.Num()) ? 
                        Agent->CurrentState.Features[i] : 0.0f;
                }
                else
                {
                    // Accumulating traces
                    Agent->EligibilityTraces[i] = Lambda * DiscountFactor * Agent->EligibilityTraces[i];
                    if (i < Agent->CurrentState.Features.Num())
                    {
                        Agent->EligibilityTraces[i] += Agent->CurrentState.Features[i];
                    }
                }
                
                // Update weight using eligibility trace
                Agent->ValueWeights[i] += CurrentLearningRate * Delta * Agent->EligibilityTraces[i];
            }
        }
        
        // Adjust learning rate if adaptive learning is enabled
        ParamManager->AdjustLearningRate(Delta);
    }
    else
    {
        // Legacy update without eligibility traces
        for (int32 i = 0; i < Agent->ValueWeights.Num(); i++)
        {
            if (i < Agent->CurrentState.Features.Num())
            {
                Agent->ValueWeights[i] += LearningRate * Delta * Agent->CurrentState.Features[i];
            }
        }
    }
}

float ARLAgentManager::CalculateTDError(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState) const
{
    if (!Agent) return 0.0f;
    
    // Calculate current state value
    float V_current = ComputeValue(Agent->ValueWeights, Agent->CurrentState.Features);
    
    // Calculate next state value
    float V_next = ComputeValue(Agent->ValueWeights, NextState);
    
    // Return TD error
    return Reward + DiscountFactor * V_next - V_current;
}