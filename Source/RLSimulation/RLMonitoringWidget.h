#pragma once
#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "Components/TextBlock.h"
#include "Components/ProgressBar.h"
#include "Components/VerticalBox.h"
#include "Components/CanvasPanel.h"
#include "RLMonitoringWidget.generated.h"

class URLAgentComponent;
class URLParameterManager;

/**
 * Data series structure for visualization
 */
USTRUCT(BlueprintType)
struct FRLDataSeries
{
    GENERATED_BODY()
    
    UPROPERTY(BlueprintReadWrite, Category="RL|Monitoring")
    FString Name;
    
    UPROPERTY(BlueprintReadWrite, Category="RL|Monitoring")
    TArray<float> Values;
    
    UPROPERTY(BlueprintReadWrite, Category="RL|Monitoring")
    FLinearColor Color = FLinearColor::White;
    
    FRLDataSeries() {}
    
    FRLDataSeries(const FString& InName, const TArray<float>& InValues, const FLinearColor& InColor)
        : Name(InName), Values(InValues), Color(InColor)
    {}
};

/**
 * Item for eligibility trace display
 */
USTRUCT(BlueprintType)
struct FRLEligibilityTraceItem
{
    GENERATED_BODY()
    
    UPROPERTY(BlueprintReadWrite, Category="RL|Monitoring")
    FString StateKey;
    
    UPROPERTY(BlueprintReadWrite, Category="RL|Monitoring")
    float Value;
    
    FRLEligibilityTraceItem() : Value(0.0f) {}
    FRLEligibilityTraceItem(const FString& InKey, float InValue) 
        : StateKey(InKey), Value(InValue) {}
};

/**
 * Monitoring widget for reinforcement learning agents
 */
UCLASS()
class RLSIMULATION_API URLMonitoringWidget : public UUserWidget
{
    GENERATED_BODY()
    
public:
    URLMonitoringWidget(const FObjectInitializer& ObjectInitializer);
    
    // Update methods
    UFUNCTION(BlueprintCallable, Category="RL|Monitoring")
    void UpdateAgentData(URLAgentComponent* Agent);
    
    UFUNCTION(BlueprintCallable, Category="RL|Monitoring")
    void UpdateLearningParams(URLParameterManager* Params);
    
    // Config options
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Monitoring")
    int32 HistoryLength = 100;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Monitoring")
    bool bShowPerformanceMetrics = true;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Monitoring")
    FLinearColor RewardGraphColor = FLinearColor(0.2f, 0.6f, 0.9f, 1.0f);
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Monitoring")
    FLinearColor ValueGraphColor = FLinearColor(0.9f, 0.4f, 0.1f, 1.0f);
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL|Monitoring")
    FLinearColor LearningRateColor = FLinearColor(0.2f, 0.8f, 0.2f, 1.0f);
    
protected:
    // Override UUserWidget interface
    virtual void NativeConstruct() override;
    virtual int32 NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, 
                            const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements, 
                            int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;
    
    // Helper method for chart drawing
    static void DrawChart(const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, 
                          FSlateWindowElementList& OutDrawElements, int32& LayerId, 
                          const TArray<FRLDataSeries>& DataSeries, FText Title);
    
    virtual void NativeTick(const FGeometry& MyGeometry, float InDeltaTime) override;
    
    // Widget components - bindable to blueprint
    UPROPERTY(meta=(BindWidget))
    UTextBlock* LearningRateText;
    
    UPROPERTY(meta=(BindWidget))
    UProgressBar* ExplorationBar;
    
    UPROPERTY(meta=(BindWidget))
    UTextBlock* StatsText;
    
    UPROPERTY(meta=(BindWidget))
    UTextBlock* PerformanceText;
    
    UPROPERTY(meta=(BindWidget))
    UCanvasPanel* RewardChartPanel;
    
    UPROPERTY(meta=(BindWidget))
    UCanvasPanel* ValueChartPanel;
    
    UPROPERTY(meta=(BindWidget))
    UVerticalBox* EligibilityTracesBox;
    
private:
    // Data storage
    TArray<float> RewardHistory;
    TArray<float> ValueHistory;
    TArray<float> LearningRateHistory;
    TArray<FRLEligibilityTraceItem> EligibilityTraces;
    
    // Current state
    float CurrentLearningRate = 0.1f;
    float CurrentExploration = 1.0f;
    int32 CurrentTargetsFound = 0;
    int32 CurrentStepsTaken = 0;
    float CurrentCumulativeReward = 0.0f;
    
    // Performance tracking
    double UpdateTimeAccumulator = 0.0;
    int32 UpdateCount = 0;
    double LastUpdateTime = 0.0;
    
    // Helper methods
    void AddDataPoint(TArray<float>& History, float Value);
    void UpdateEligibilityTracesUI();
    void UpdateStatsText();
    void UpdatePerformanceText();
    
};

