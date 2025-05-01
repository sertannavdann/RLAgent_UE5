#include "RLSimulationGameMode.h"
#include "RLMonitoringWidget.h"
#include "RLAgentManager.h"
#include "RLAgentComponent.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"

ARLSimulationGameMode::ARLSimulationGameMode()
{
    // Enable tick for the game mode
    PrimaryActorTick.bCanEverTick = true;
}

void ARLSimulationGameMode::BeginPlay()
{
    Super::BeginPlay();
    
    // Try to find agent manager in the level if not already set
    if (!AgentManager)
    {
        // First try to find an existing agent manager
        for (TActorIterator<ARLAgentManager> It(GetWorld()); It; ++It)
        {
            AgentManager = *It;
            LogDebugInfo(FString::Printf(TEXT("Found AgentManager: %s"), *AgentManager->GetName()));
            break;
        }
        
        // If still not found, create a new one
        if (!AgentManager)
        {
            FActorSpawnParameters SpawnParams;
            SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AgentManager = GetWorld()->SpawnActor<ARLAgentManager>(ARLAgentManager::StaticClass(), SpawnParams);
            if (AgentManager)
            {
                LogDebugInfo(FString::Printf(TEXT("Created new AgentManager: %s"), *AgentManager->GetName()));
            }
            else
            {
                LogDebugInfo(TEXT("Failed to create AgentManager"));
            }
        }
    }
    
    // Create monitoring widget
    if (MonitoringWidgetClass)
    {
        MonitoringWidget = CreateWidget<URLMonitoringWidget>(GetWorld(), MonitoringWidgetClass);
        if (MonitoringWidget)
        {
            MonitoringWidget->AddToViewport();
            LogDebugInfo("Successfully created and added monitoring widget to viewport");
        }
        else
        {
            LogDebugInfo("Failed to create monitoring widget");
        }
    }
    else
    {
        LogDebugInfo("MonitoringWidgetClass not set");
    }
}

void ARLSimulationGameMode::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    
    // Only update monitoring on certain frames for performance
    UpdateCounter++;
    if (UpdateCounter >= MonitoringUpdateInterval)
    {
        UpdateMonitoring();
        UpdateCounter = 0;
    }
}

void ARLSimulationGameMode::SetAgentManager(ARLAgentManager* InAgentManager)
{
    AgentManager = InAgentManager;
    LogDebugInfo(FString::Printf(TEXT("AgentManager set to: %s"), 
        AgentManager ? *AgentManager->GetName() : TEXT("None")));
}

void ARLSimulationGameMode::UpdateMonitoring() const
{
    if (!MonitoringWidget)
        return;
    
    // Get an active agent for monitoring
    if (URLAgentComponent* ActiveAgent = GetActiveAgent())
    {
        // Update monitoring with agent data
        MonitoringWidget->UpdateAgentData(ActiveAgent);
        
        // Update parameter info if available
        if (ActiveAgent->ParameterManager)
        {
            MonitoringWidget->UpdateLearningParams(ActiveAgent->ParameterManager);
        }
        else if (AgentManager && AgentManager->GlobalParameterManager)
        {
            // Fall back to global parameter manager
            MonitoringWidget->UpdateLearningParams(AgentManager->GlobalParameterManager);
        }
    }
}

URLAgentComponent* ARLSimulationGameMode::GetActiveAgent() const
{
    // Try to get agent from the manager
    if (AgentManager && AgentManager->Agents.Num() > 0)
    {
        return AgentManager->Agents[0];
    }
    
    // Otherwise, try to find any agent component in the level
    AActor* FoundActor = nullptr;
    URLAgentComponent* FoundComponent = nullptr;
    
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (URLAgentComponent* Component = Cast<URLAgentComponent>(It->GetComponentByClass(URLAgentComponent::StaticClass())))
        {
            return Component;
        }
    }
    
    return nullptr;
}

void ARLSimulationGameMode::LogDebugInfo(const FString& Message) const
{
    if (bEnableDebugLogging)
    {
        UE_LOG(LogTemp, Display, TEXT("[RLSimulation] %s"), *Message);
    }
}