#include "RLMonitoringWidget.h"
#include "RLAgentComponent.h"
#include "RLParameterManager.h"
#include "Blueprint/WidgetTree.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "Styling/SlateStyleRegistry.h"
#include "Styling/AppStyle.h"

URLMonitoringWidget::URLMonitoringWidget(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer),
      LearningRateText(nullptr),
      ExplorationBar(nullptr),
      StatsText(nullptr),
      PerformanceText(nullptr),
      RewardChartPanel(nullptr),
      ValueChartPanel(nullptr),
      EligibilityTracesBox(nullptr)
{
    // Initialize history arrays
    RewardHistory.Init(0.0f, HistoryLength);
    ValueHistory.Init(0.0f, HistoryLength);
    LearningRateHistory.Init(0.1f, HistoryLength);
}

void URLMonitoringWidget::NativeConstruct()
{
    Super::NativeConstruct();
    
    // Set up performance tracking
    LastUpdateTime = FPlatformTime::Seconds();
    
    // Initialize UI
    UpdateStatsText();
    UpdatePerformanceText();
}

int32 URLMonitoringWidget::NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, 
                                      const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements, 
                                      int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
    // Paint the parent first
    int32 MaxLayer = Super::NativePaint(Args, AllottedGeometry, MyCullingRect, OutDrawElements, LayerId, InWidgetStyle, bParentEnabled);
    
    // Paint reward chart if the panel exists and is visible
    if (RewardChartPanel && RewardChartPanel->GetVisibility() != ESlateVisibility::Collapsed)
    {
        // Get panel's cached geometry directly
        const FGeometry& PanelGeometry = RewardChartPanel->GetCachedGeometry();
        
        // Create reward data series
        TArray<FRLDataSeries> RewardData;
        RewardData.Add(FRLDataSeries("Reward", RewardHistory, RewardGraphColor));
        
        // Draw the chart
        DrawChart(PanelGeometry, MyCullingRect, OutDrawElements, MaxLayer, RewardData, FText::FromString("Reward History"));
    }
    
    // Paint value chart if the panel exists and is visible
    if (ValueChartPanel && ValueChartPanel->GetVisibility() != ESlateVisibility::Collapsed)
    {
        // Get panel's cached geometry directly
        const FGeometry& PanelGeometry = ValueChartPanel->GetCachedGeometry();
        
        // Create value data series
        TArray<FRLDataSeries> ValueData;
        ValueData.Add(FRLDataSeries("Value", ValueHistory, ValueGraphColor));
        ValueData.Add(FRLDataSeries("Learning Rate", LearningRateHistory, LearningRateColor));
        
        // Draw the chart
        DrawChart(PanelGeometry, MyCullingRect, OutDrawElements, MaxLayer, ValueData, FText::FromString("Value Function"));
    }
    
    return MaxLayer;
}

void URLMonitoringWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
    Super::NativeTick(MyGeometry, InDeltaTime);
    
    // Reset performance stats periodically
    double CurrentTime = FPlatformTime::Seconds();
    if ((CurrentTime - LastUpdateTime) > 5.0 && UpdateCount > 0)
    {
        UpdateTimeAccumulator = 0.0;
        UpdateCount = 0;
        LastUpdateTime = CurrentTime;
        UpdatePerformanceText();
    }
}

void URLMonitoringWidget::UpdateAgentData(URLAgentComponent* Agent)
{
    if (!Agent) return;
    
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Update Monitoring UI");
    
    // Measure update time for performance tracking
    double StartTime = FPlatformTime::Seconds();
    
    // Update reward history
    if (Agent->RewardHistory.Num() > 0)
    {
        // Copy from agent if available
        for (int32 i = 0; i < FMath::Min(Agent->RewardHistory.Num(), RewardHistory.Num()); i++)
        {
            RewardHistory[i] = Agent->RewardHistory[i];
        }
    }
    else
    {
        // Add current reward delta
        AddDataPoint(RewardHistory, Agent->CumulativeReward - CurrentCumulativeReward);
    }
    
    // Update value function estimate
    float CurrentValue = 0.0f;
    if (Agent->CurrentState.Features.Num() > 0 && Agent->ValueWeights.Num() > 0)
    {
        for (int32 i = 0; i < FMath::Min(Agent->ValueWeights.Num(), Agent->CurrentState.Features.Num()); i++)
        {
            CurrentValue += Agent->ValueWeights[i] * Agent->CurrentState.Features[i];
        }
    }
    AddDataPoint(ValueHistory, CurrentValue);
    
    // Update agent stats
    CurrentTargetsFound = Agent->TargetsFound;
    CurrentStepsTaken = Agent->StepsTaken;
    CurrentCumulativeReward = Agent->CumulativeReward;
    
    // Update eligibility traces if parameter manager is available
    if (Agent->ParameterManager)
    {
        // Clear previous traces
        EligibilityTraces.Empty();
        
        // Get top traces from parameter manager
        TArray<FString> TraceKeys;
        TArray<float> TraceValues;
        Agent->ParameterManager->GetTopEligibilityTraces(TraceKeys, TraceValues, 10);
        
        // Store traces
        for (int32 i = 0; i < FMath::Min(TraceKeys.Num(), TraceValues.Num()); i++)
        {
            EligibilityTraces.Add(FRLEligibilityTraceItem(TraceKeys[i], TraceValues[i]));
        }
        
        // Update UI
        UpdateEligibilityTracesUI();
    }
    
    // Update text displays
    UpdateStatsText();
    
    // Refresh chart panels
    if (RewardChartPanel)
    {
        RewardChartPanel->InvalidateLayoutAndVolatility();
    }
    
    if (ValueChartPanel)
    {
        ValueChartPanel->InvalidateLayoutAndVolatility();
    }
    
    // Track performance
    double EndTime = FPlatformTime::Seconds();
    UpdateTimeAccumulator += (EndTime - StartTime);
    UpdateCount++;
    UpdatePerformanceText();
}

void URLMonitoringWidget::UpdateLearningParams(URLParameterManager* Params)
{
    if (!Params) return;
    
    // Update learning rate and exploration rate
    CurrentLearningRate = Params->GetCurrentLearningRate();
    CurrentExploration = FMath::Clamp(Params->GetCurrentLearningRate() / Params->InitialLearningRate, 0.0f, 1.0f);
    
    // Update learning rate history
    AddDataPoint(LearningRateHistory, CurrentLearningRate);
    
    // Update UI
    if (LearningRateText)
    {
        LearningRateText->SetText(FText::Format(NSLOCTEXT("RLMonitoring", "LearningRateFormat", 
            "Learning Rate: {0}"), FText::AsNumber(CurrentLearningRate)));
    }
    
    if (ExplorationBar)
    {
        ExplorationBar->SetPercent(CurrentExploration);
    }
    
    // Refresh value chart
    if (ValueChartPanel)
    {
        ValueChartPanel->InvalidateLayoutAndVolatility();
    }
}

void URLMonitoringWidget::DrawChart(const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, 
                                   FSlateWindowElementList& OutDrawElements, int32& LayerId, 
                                   const TArray<FRLDataSeries>& DataSeries, FText Title)
{
    if (DataSeries.Num() == 0)
        return;

    // Get drawing dimensions
    const FVector2D Size = AllottedGeometry.GetLocalSize();
    if (Size.IsZero())
        return;
    
    // Chart dimensions with padding
    const float Padding = 20.0f;
    const FVector2D ChartPos(Padding, Padding);
    const FVector2D ChartSize = Size - (ChartPos * 2.0f);
    
    if (ChartSize.X <= 0 || ChartSize.Y <= 0)
        return;
    
    // Draw title
    FSlateDrawElement::MakeText(
        OutDrawElements,
        ++LayerId,
        AllottedGeometry.ToPaintGeometry(FVector2D(200, 20), FSlateLayoutTransform(FVector2D(5, 5))),
        Title,
        // Use AppStyle instead of FCoreStyle for UE5.5 compatibility
        FAppStyle::Get().GetFontStyle("NormalFont"),
        ESlateDrawEffect::None,
        FLinearColor::White
    );
    
    // Find value range
    float MinValue = FLT_MAX;
    float MaxValue = -FLT_MAX;
    
    for (const FRLDataSeries& Series : DataSeries)
    {
        for (float Value : Series.Values)
        {
            MinValue = FMath::Min(MinValue, Value);
            MaxValue = FMath::Max(MaxValue, Value);
        }
    }
    
    // Add padding to range
    float ValueRange = MaxValue - MinValue;
    if (ValueRange < KINDA_SMALL_NUMBER)
    {
        MinValue -= 0.5f;
        MaxValue += 0.5f;
        ValueRange = 1.0f;
    }
    else
    {
        MinValue -= ValueRange * 0.1f;
        MaxValue += ValueRange * 0.1f;
        ValueRange = MaxValue - MinValue;
    }
    
    // Draw each data series
    for (const FRLDataSeries& Series : DataSeries)
    {
        if (Series.Values.Num() < 2) 
            continue;
        
        // Draw lines connecting points
        TArray<FVector2D> Points;
        for (int32 i = 0; i < Series.Values.Num(); i++)
        {
            float XPos = ChartPos.X + (static_cast<float>(i) / (Series.Values.Num() - 1)) * ChartSize.X;
            float YPos = ChartPos.Y + ChartSize.Y - ((Series.Values[i] - MinValue) / ValueRange) * ChartSize.Y;
            Points.Add(FVector2D(XPos, YPos));
        }
        
        // Draw the line segments properly for UE5.5
        FSlateDrawElement::MakeLines(
            OutDrawElements,
            ++LayerId,
            AllottedGeometry.ToPaintGeometry(),
            Points,
            ESlateDrawEffect::None,
            Series.Color,
            true,
            1.5f
        );
        
        // Draw series name
        FVector2D TextPos(Points.Last().X - 60.0f, Points.Last().Y - 15.0f);
        TextPos.X = FMath::Clamp(TextPos.X, ChartPos.X, ChartPos.X + ChartSize.X - 60.0f);
        TextPos.Y = FMath::Clamp(TextPos.Y, ChartPos.Y, ChartPos.Y + ChartSize.Y - 15.0f);
        
        FSlateDrawElement::MakeText(
            OutDrawElements,
            ++LayerId,
            AllottedGeometry.ToPaintGeometry(FVector2D(60.0f, 15.0f), FSlateLayoutTransform(TextPos)),
            FText::FromString(Series.Name),
            FAppStyle::Get().GetFontStyle("SmallFont"),
            ESlateDrawEffect::None,
            Series.Color
        );
    }
}

void URLMonitoringWidget::AddDataPoint(TArray<float>& History, float Value)
{
    if (History.Num() > 0)
    {
        // Shift values left (remove oldest)
        for (int32 i = 0; i < History.Num() - 1; i++)
        {
            History[i] = History[i + 1];
        }
        
        // Add new value at the end
        History[History.Num() - 1] = Value;
    }
}

void URLMonitoringWidget::UpdateEligibilityTracesUI()
{
    if (!EligibilityTracesBox) 
        return;
    
    // Clear existing entries
    EligibilityTracesBox->ClearChildren();
    
    // Add header
    if (EligibilityTraces.Num() > 0)
    {
        // Create header text
        UTextBlock* HeaderText = WidgetTree->ConstructWidget<UTextBlock>(UTextBlock::StaticClass());
        HeaderText->SetText(FText::FromString("Top Eligibility Traces"));
        HeaderText->SetColorAndOpacity(FSlateColor(FLinearColor(1.0f, 1.0f, 0.8f)));
        EligibilityTracesBox->AddChild(HeaderText);
        
        // Add up to 5 trace entries
        for (int32 i = 0; i < FMath::Min(EligibilityTraces.Num(), 5); i++)
        {
            UTextBlock* TraceText = WidgetTree->ConstructWidget<UTextBlock>(UTextBlock::StaticClass());
            TraceText->SetText(FText::Format(NSLOCTEXT("RLMonitoring", "TraceFormat", 
                "{0}: {1}"), FText::FromString(EligibilityTraces[i].StateKey), 
                FText::AsNumber(EligibilityTraces[i].Value)));
                
            EligibilityTracesBox->AddChild(TraceText);
        }
    }
}

void URLMonitoringWidget::UpdateStatsText()
{
    if (!StatsText) 
        return;
    
    StatsText->SetText(FText::Format(NSLOCTEXT("RLMonitoring", "StatsFormat", 
        "Targets Found: {0} | Steps Taken: {1} | Cumulative Reward: {2}"),
        CurrentTargetsFound,
        CurrentStepsTaken,
        FMath::RoundToFloat(CurrentCumulativeReward)));
}

void URLMonitoringWidget::UpdatePerformanceText()
{
    if (!PerformanceText || !bShowPerformanceMetrics) 
        return;
    
    float AvgUpdateTime = (UpdateCount > 0) ? 
        (UpdateTimeAccumulator / UpdateCount) * 1000.0f : 0.0f; // Convert to ms
    
    #if PLATFORM_MAC
        PerformanceText->SetText(FText::Format(NSLOCTEXT("RLMonitoring", "PerformanceFormatMac", 
            "Avg Update: {0}ms | Updates: {1} | Memory: {2}MB"), 
            FMath::RoundToFloat(AvgUpdateTime), 
            UpdateCount,
            FMath::RoundToFloat(FPlatformMemory::GetStats().UsedPhysical / (1024.0f * 1024.0f))));
    #else
        PerformanceText->SetText(FText::Format(NSLOCTEXT("RLMonitoring", "PerformanceFormat", 
            "Avg Update: {0}ms | Updates: {1}"), 
            FMath::RoundToFloat(AvgUpdateTime), 
            UpdateCount));
    #endif
}