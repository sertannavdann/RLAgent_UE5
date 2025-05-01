// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameMode.h"
#include "RLSimulationGameMode.generated.h"

/**
 * 
 */
UCLASS()
class RLSIMULATION_API ARLSimulationGameMode : public AGameMode
{
	GENERATED_BODY()
	
    UPROPERTY(EditDefaultsOnly, Category="UI")
    TSubclassOf<UUserWidget> RLMonitoringWidgetClass;

    UPROPERTY()
    URLMonitoringWidget* MonitoringWidget;
};
