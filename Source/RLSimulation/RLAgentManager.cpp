#include "RLAgentManager.h"
#include "RLAgentComponent.h"
#include "DrawDebugHelpers.h"

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
    EpsilonDecay = 0.995f;    // Slower decay
    LearningRate = 0.1f;      // Simple learning rate
    DiscountFactor = 0.9f;    // Standard discount
    
    // Update NumStateFeatures to 6 for full 3D
    NumStateFeatures = 6;
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
    
    // Spawn a debug marker at the target position
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

        // 5. Simple TD Update
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
    
    // Draw the training area - 3D sphere
    DrawDebugSphere(GetWorld(), FVector::ZeroVector, SphereRadius, 24, FColor::Green, false, -1, 0, 2.0f);
    
    // Draw the target object
    DrawDebugSphere(GetWorld(), TargetObjectPosition, 50.f, 8, FColor::Red, false, -1, 0, 5.0f);
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
            CurrentLoc += CurrentRot.Vector() * 10.f;
            break;
        case 1: // Turn left
            CurrentRot.Yaw -= 1.5f;
            break;
        case 2: // Turn right
            CurrentRot.Yaw += 1.5f;
            break;
        case 3: // Move up
            CurrentLoc.Z += 10.f;
            break;
        case 4: // Move down
            CurrentLoc.Z -= 10.f;
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
    // Get current state value
    const float V_Current = ComputeValue(Agent->ValueWeights, Agent->CurrentState.Features);
    
    // Get next state value
        const float V_Next = ComputeValue(Agent->ValueWeights, NextState);
    
    // Calculate TD error
    const float Delta = Reward + DiscountFactor * V_Next - V_Current;

    // Update weights directly
    for (int32 i = 0; i < Agent->ValueWeights.Num(); i++)
    {
        Agent->ValueWeights[i] += LearningRate * Delta * Agent->CurrentState.Features[i];
    }
}