#include "RLAgentManager.h"
#include "RLAgentComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"
#include "Kismet/KismetMathLibrary.h"

ARLAgentManager::ARLAgentManager()
{
    PrimaryActorTick.bCanEverTick = true;
    
    // Set up the target object position - this is what agents will look for
    TargetObjectPosition = FVector(FMath::RandRange(-1000.f, 1000.f),
                                  FMath::RandRange(-1000.f, 1000.f),
                                  0.f);
    
    // Sphere radius represents the training area
    SphereRadius = 1500.f;
    
    // Simplified learning parameters
    InitialEpsilon = 0.3f;    // Less exploration
    EpsilonDecay = 0.999f;    // Slower decay
    LearningRate = 0.1f;      // Simple learning rate
    DiscountFactor = 0.9f;    // Standard discount
}

int32 ARLAgentManager::RegisterAgent(URLAgentComponent* Agent)
{
    Agents.Add(Agent);
    return Agents.Num() - 1;
}

void ARLAgentManager::StartTraining()
{
    CurrentEpsilon = InitialEpsilon;
    
    // Spawn a debug marker at the target position
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, true);
}

// Respawn the target in a new random position within the sphere
void ARLAgentManager::RespawnTarget()
{
    // Clear old target debug markers
    FlushPersistentDebugLines(GetWorld());
    
    // Generate new random position within sphere
    float Radius = FMath::RandRange(0.f, SphereRadius * 0.8f);
    float Angle = FMath::RandRange(0.f, 2.f * PI);
    
    TargetObjectPosition = FVector(
        Radius * FMath::Cos(Angle),
        Radius * FMath::Sin(Angle),
        0.f
    );
    
    // Draw the new target
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, true);
}

void ARLAgentManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (Agents.Num() == 0) return;

    for (URLAgentComponent* Agent : Agents)
    {
        // 1. Observe current state
        TArray<float> CurrentFeatures = GetStateFeatures(Agent);
        Agent->CurrentState.Features = CurrentFeatures;

        // 2. Choose action: Simplified to 3 options (forward, left, right)
        int32 Action = 0;
        if (FMath::FRand() < CurrentEpsilon)
        {
            // Random exploration - just pick any action
            Action = FMath::RandRange(0, 2);
        }
        else
        {
            // Simplified action selection - just evaluate each action directly
            float BestValue = -FLT_MAX;
            for (int32 PotentialAction = 0; PotentialAction < 3; PotentialAction++)
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

        // 5. Simple TD Update without eligibility traces
        TDUpdate(Agent, Reward, NextFeatures);

        // 6. Update state and bookkeeping
        Agent->CurrentState.Features = NextFeatures;
        Agent->CumulativeReward += Reward;
        Agent->PreviousLocation = Agent->GetOwner()->GetActorLocation();

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
            RespawnTarget();
        }
    }

    // Reduce exploration over time (more slowly)
    CurrentEpsilon = FMath::Max(0.05f, CurrentEpsilon * EpsilonDecay);
    
    // Draw the training area
    DrawDebugSphere(GetWorld(), FVector::ZeroVector, SphereRadius, 24, FColor::Green, false, -1, 0, 2.0f);
    
    // Draw the target object
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, false, -1, 0, 5.0f);
}

TArray<float> ARLAgentManager::GetStateFeatures(URLAgentComponent* Agent)
{
    // Simplified state: just position and direction
    FVector Loc = Agent->GetOwner()->GetActorLocation();
    FVector Forward = Agent->GetOwner()->GetActorForwardVector();

    return {
        Loc.X / 1000.f,       // Normalized X position
        Loc.Y / 1000.f,       // Normalized Y position
        Forward.X,            // Direction X
        Forward.Y             // Direction Y
    };
}

TArray<float> ARLAgentManager::GetPotentialState(URLAgentComponent* Agent, int32 Action)
{
    // Get current state without modifying the actor
    FVector CurrentLoc = Agent->GetOwner()->GetActorLocation();
    FRotator CurrentRot = Agent->GetOwner()->GetActorRotation();

    // Simulate action
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
    }

    // Return simplified features [X, Y, ForwardX, ForwardY]
    FVector SimulatedForward = FRotator(0, CurrentRot.Yaw, 0).Vector();
    return {
        CurrentLoc.X / 1000.f,
        CurrentLoc.Y / 1000.f,
        SimulatedForward.X,
        SimulatedForward.Y
    };
}

TArray<float> ARLAgentManager::GetNextState(URLAgentComponent* Agent, int32 Action)
{
    FVector Loc = Agent->GetOwner()->GetActorLocation();
    
    // Boundary check - keep agent within the sphere
    if (Loc.Size() > SphereRadius)
    {
        // Teleport back toward the center if agent goes outside the sphere
        FVector Direction = -Loc.GetSafeNormal();
        Loc = Direction * (SphereRadius * 0.8f);
        Agent->GetOwner()->SetActorLocation(Loc);
    }

    // Apply action directly
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
    }
    
    Agent->GetOwner()->SetActorLocation(Loc);
    return GetStateFeatures(Agent);
}

float ARLAgentManager::CalculateReward(URLAgentComponent* Agent, const TArray<float>& NextState)
{
    // The main objective is to find the target object
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

void ARLAgentManager::TDUpdate(URLAgentComponent* Agent, float Reward, const TArray<float>& NextState)
{
    // Get current state value
    float V_current = ComputeValue(Agent->ValueWeights, Agent->CurrentState.Features);
    
    // Get next state value
    float V_next = ComputeValue(Agent->ValueWeights, NextState);
    
    // Calculate TD error (simpler version without eligibility traces)
    float Delta = Reward + DiscountFactor * V_next - V_current;

    // Update weights directly
    for (int32 i = 0; i < Agent->ValueWeights.Num(); i++)
    {
        Agent->ValueWeights[i] += LearningRate * Delta * Agent->CurrentState.Features[i];
    }
}
