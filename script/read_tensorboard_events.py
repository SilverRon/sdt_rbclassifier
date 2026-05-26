#!/usr/bin/env python3
"""
TensorBoard 이벤트 파일 읽기 예제
"""

from tensorboard.backend.event_processing import event_accumulator
import os
from pathlib import Path

# 이벤트 파일 경로
event_file = "/data/data1/rb_classification_meta/output/logs/v2A_bal_bs512_lr3e4/version_0/events.out.tfevents.1768791020.proton.snu.ac.kr.3891209.0"

print("="*80)
print("TENSORBOARD 이벤트 파일 분석")
print("="*80)
print(f"파일: {Path(event_file).name}")
print(f"크기: {os.path.getsize(event_file):,} bytes")
print("="*80)

# EventAccumulator 생성
# size_guidance: 각 태그별로 메모리에 유지할 이벤트 수
ea = event_accumulator.EventAccumulator(
    str(Path(event_file).parent),
    size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,  # 0 = 모두 로드
        event_accumulator.HISTOGRAMS: 1,
    }
)

# 데이터 로드
print("\n데이터 로딩 중...")
ea.Reload()
print("✓ 로딩 완료\n")

# 1. 사용 가능한 태그 확인
print("="*80)
print("1. 사용 가능한 태그 (Tags)")
print("="*80)

tags = ea.Tags()
print(f"\n📊 Scalars (스칼라 메트릭): {len(tags['scalars'])}개")
for tag in sorted(tags['scalars']):
    print(f"  - {tag}")

if tags['histograms']:
    print(f"\n📈 Histograms: {len(tags['histograms'])}개")
    for tag in tags['histograms']:
        print(f"  - {tag}")

if tags['images']:
    print(f"\n🖼️  Images: {len(tags['images'])}개")
    for tag in tags['images']:
        print(f"  - {tag}")

# 2. Scalar 데이터 상세 분석
print("\n" + "="*80)
print("2. Scalar 메트릭 상세 정보")
print("="*80)

for tag in sorted(tags['scalars']):
    events = ea.Scalars(tag)
    if events:
        print(f"\n📊 {tag}")
        print(f"   총 이벤트 수: {len(events)}")
        print(f"   첫 값: {events[0].value:.6f} (step {events[0].step}, time {events[0].wall_time:.2f})")
        if len(events) > 1:
            print(f"   마지막 값: {events[-1].value:.6f} (step {events[-1].step})")
            print(f"   최소값: {min(e.value for e in events):.6f}")
            print(f"   최대값: {max(e.value for e in events):.6f}")

# 3. 학습 곡선 데이터 추출
print("\n" + "="*80)
print("3. 학습 곡선 데이터 (최근 10개 step)")
print("="*80)

# Train/Val Loss
if 'train_loss' in tags['scalars']:
    train_loss_events = ea.Scalars('train_loss')
    print(f"\n📉 Train Loss (총 {len(train_loss_events)}개 기록)")
    print("   Step | Value")
    print("   -----|-------")
    for event in train_loss_events[-10:]:
        print(f"   {event.step:4d} | {event.value:.6f}")

if 'val_loss' in tags['scalars']:
    val_loss_events = ea.Scalars('val_loss')
    print(f"\n📉 Val Loss (총 {len(val_loss_events)}개 기록)")
    print("   Step | Value")
    print("   -----|-------")
    for event in val_loss_events[-10:]:
        print(f"   {event.step:4d} | {event.value:.6f}")

# 4. 최종 성능 요약
print("\n" + "="*80)
print("4. 최종 성능 요약 (마지막 epoch)")
print("="*80)

metrics_of_interest = [
    'train_loss', 'train_acc',
    'val_loss', 'val_acc', 'val_f1', 'val_auroc',
    'test_loss', 'test_acc', 'test_f1', 'test_prec', 'test_rec', 'test_auroc'
]

print("\n| Metric | Value | Step |")
print("|--------|-------|------|")
for metric in metrics_of_interest:
    if metric in tags['scalars']:
        events = ea.Scalars(metric)
        if events:
            last_event = events[-1]
            print(f"| {metric:15s} | {last_event.value:7.4f} | {last_event.step:4d} |")

# 5. 데이터 내보내기 (CSV)
print("\n" + "="*80)
print("5. CSV로 내보내기")
print("="*80)

import csv

output_dir = Path("/data/data1/rb_classification_meta/output/logs/v2A_bal_bs512_lr3e4/version_0")
csv_file = output_dir / "metrics_export.csv"

# 모든 scalar를 CSV로 저장
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'step', 'value', 'wall_time'])
    
    for tag in sorted(tags['scalars']):
        events = ea.Scalars(tag)
        for event in events:
            writer.writerow([tag, event.step, event.value, event.wall_time])

print(f"✓ CSV 파일 생성: {csv_file}")
print(f"  총 {sum(len(ea.Scalars(tag)) for tag in tags['scalars'])} 개 데이터 포인트 저장")

# 6. Epoch별 요약
print("\n" + "="*80)
print("6. Epoch별 학습 진행 상황")
print("="*80)

if 'train_loss' in tags['scalars'] and 'val_loss' in tags['scalars']:
    train_loss = ea.Scalars('train_loss')
    val_loss = ea.Scalars('val_loss')
    train_acc = ea.Scalars('train_acc') if 'train_acc' in tags['scalars'] else []
    val_acc = ea.Scalars('val_acc') if 'val_acc' in tags['scalars'] else []
    
    print("\n| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |")
    print("|-------|------------|-----------|----------|---------|")
    
    # Epoch은 step과 동일하다고 가정
    for i in range(min(len(train_loss), len(val_loss))):
        epoch = train_loss[i].step
        tl = train_loss[i].value
        vl = val_loss[i].value
        ta = train_acc[i].value if i < len(train_acc) else 0
        va = val_acc[i].value if i < len(val_acc) else 0
        print(f"| {epoch:5d} | {tl:10.6f} | {ta:9.4f} | {vl:8.6f} | {va:7.4f} |")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n💡 TensorBoard로 시각화하려면:")
print(f"   tensorboard --logdir=/data/data1/rb_classification_meta/output/logs")
print(f"   그리고 브라우저에서 http://lyman.snu.ac.kr:6006/ 접속")
print("\n💡 CSV 파일 확인:")
print(f"   cat {csv_file}")
print("="*80)
