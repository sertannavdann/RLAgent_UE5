#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameMode.h"
#include "RLSimulationGameMode.generated.h"

class URLMonitoringWidget;
class ARLAgentManager;
class URLAgentComponent;

/**
 * Game Mode for Reinforcement Learning Simulation
 * Handles initialization and updating of monitoring widgets
 */
UCLASS()
class RLSIMULATION_API ARLSimulationGameMode : public AGameMode
{
    GENERATED_BODY()
    
public:
    ARLSimulationGameMode();

    // Called when the game starts or when spawned
    virtual void BeginPlay() override;
    
    // Called every frame
    virtual void Tick(float DeltaTime) override;
    
    // Blueprint event to find and link the agent manager
    UFUNCTION(BlueprintCallable, Category="Reinforcement Learning")
    void SetAgentManager(ARLAgentManager* InAgentManager);
    
    // Reference to the monitoring widget class
    UPROPERTY(EditDefaultsOnly, Category="UI")
    TSubclassOf<UUserWidget> MonitoringWidgetClass;

    // Determines how often the monitoring widget updates (frames)
    UPROPERTY(EditDefaultsOnly, Category="UI")
    int32 MonitoringUpdateInterval = 5;
    
    // Enable debug logging
    UPROPERTY(EditDefaultsOnly, Category="Debug")
    bool bEnableDebugLogging = false;

private:
    // The monitoring widget instance
    UPROPERTY()
    URLMonitoringWidget* MonitoringWidget;
    
    // Reference to the agent manager in the level
    UPROPERTY()
    ARLAgentManager* AgentManager;
    
    // Update counter for monitoring refresh
    int32 UpdateCounter = 0;
    
    // Updates the monitoring widget with data from agents
    void UpdateMonitoring() const;
    
    // Find and use the first active agent for monitoring
    URLAgentComponent* GetActiveAgent() const;
    
    // Log debug information if enabled
    void LogDebugInfo(const FString& Message) const;
};
