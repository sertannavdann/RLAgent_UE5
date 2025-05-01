// Fill out your copyright notice in the Description page of Project Settings.


#include "RLSimulationGameMode.h"

void RLSimulationGameMode::BeginPlay()
{
    Super::BeginPlay();
    
    // Create monitoring widget
    if (RLMonitoringWidgetClass)
    {
        MonitoringWidget = CreateWidget<URLMonitoringWidget>(GetWorld(), RLMonitoringWidgetClass);
        if (MonitoringWidget)
        {
            MonitoringWidget->AddToViewport();
        }
    }
}

// In your game tick or agent update function:
void UpdateRLMonitoring(float DeltaTime)
{
    if (MonitoringWidget && Agent)
    {
        // Update with agent data
        MonitoringWidget->UpdateAgentData(Agent);
        
        // Update learning parameters if available
        if (Agent->ParameterManager)
        {
            MonitoringWidget->UpdateLearningParams(Agent->ParameterManager);
        }
    }
}
