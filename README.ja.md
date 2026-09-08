# rustcoin3d — 産業用3D可視化エンジン

[English](README.md) · [简体中文](README.zh-cn.md) · **日本語**

Rust + wgpu による Coin3D/HOOPS 互換の3D可視化エンジン。大規模産業可視化のために設計されています：CAD インポート、リアルタイム PBR レンダリング、ノードグラフ合成、インタラクティブなシーン編集——すべてデスクトップ Studio アプリケーションに統合。

![Studio](assets/studio_shot.png)

## ハイライト

- **Rust + wgpu 30** レンダラー：クラスタ遅延 PBR、CSM シャドウ、HZB オクルージョン、TAA/SSR/SSAO
- **シーングラフ**、62種のノード型（Coin3D/Inventor スタイルの `Separator`/`Switch`/`LOD`、カメラ、ライト、注釈、マニピュレータ）
- **Blender スタイル合成器** ノードグラフ（`egui-snarl`）によるリアルタイム画像合成
- **産業用 CAD を想定**：NURBS、断面/ハッチ、GD&T/PMI、非表示線、点群（OOC）
- **デスクトップ Studio**、Model / LookDev / Compositor ワークスペース、完全な i18n（英語/簡体字中国語）
- レンダリング・照明・インポート・アニメーション・エディタ・診断を網羅する51のサンプル

## ドキュメント

| ドキュメント | 説明 |
|------|------|
| [アーキテクチャ](docs/architecture.md) | crate 依存グラフ、コア設計原則、主要データ構造、NodeData リファレンス |
| [レンダリングパイプライン](docs/rendering-pipeline.md) | フレームパイプライン全体、カリング、照明、PBR シェーディング、後処理、描画呼び出しバッチ |
| [シーングラフ](docs/scene-graph.md) | SceneGraph API、全ノード型、トラバーサルモデル、ダーティフラグ、アニメーション、シリアライズ |
| [エンジンシステム](docs/engine-system.md) | シミュレーションエンジン、時間管理、物理、センサー、フィールド接続 |
| [シェーダ](docs/shaders.md) | データ構造と性能ノートを含む完全な WGSL シェーダカタログ |
| [ギャップ分析](docs/industrial-viz-gap-analysis.md) | Coin3D/HOOPS 比較、ロードマップ、TODO チェックリスト |
| [最適化ガイド](docs/optimization-guide.md) | GPU カリング、メッシュプール、静的フレーム高速パス、LightSetTable、共有ユーティリティ |
| [変更履歴](CHANGELOG.md) | リリースノートと注目すべき変更 |

## クイックスタート

```bash
# デスクトップエディタ
cargo run -p rc3d-studio

# 全サンプルをビルド
cargo build -p rc3d-examples --examples

# 3D ファイルのインポートと表示
cargo run -p rc3d-examples --example import_viewer -- model.stl

# 適応品質ストレステスト
cargo run -p rc3d-examples --example adaptive_stress_test

# CLI エディタ（ターミナル）
cargo run -p rc3d-cli-editor
```

## アーキテクチャ

```
crates/
├── rc3d-core/       — 数学、AABB、BVH、ID 型、共有ユーティリティ（グラフ、ハッシュ、リング、ソート）
├── rc3d-fields/     — フィールド/接続システム（Coin3D スタイル）
├── rc3d-scene/      — シーングラフ（SlotMap<NodeId, NodeEntry>）、ノード型、アニメーション
├── rc3d-nodes/      — 再エクスポート（便利な crate）
├── rc3d-mesh/       — 三角メッシュ、メッシュレット生成、LOD、テッセレーション
├── rc3d-nurbs/      — NURBS 曲線と曲面
├── rc3d-actions/    — トラバーサルアクション（レイピック、バウンディングボックス、アンドゥ、イベント、交差）
├── rc3d-engine/     — シミュレーションエンジン、時間管理、物理、スケジューラ
├── rc3d-io/         — ファイルのインポート（STL、OBJ、glTF、FBX、Inventor）とエクスポート
├── rc3d-render/     — wgpu レンダラー（PBR、シャドウ、カリング、後処理、シェーダ）
├── rc3d-gizmo/      — 3D マニピュレータ（移動、回転、スケール）
├── rc3d-script/     — Rhai スクリプトエンジン
├── rc3d-pointcloud/ — 大規模点群オクトリー（OOC）
├── rc3d-pdf/        — 3D PDF エクスポート（U3D）
├── rc3d-engine-api/ — エンジンファサード（ウィンドウ、カメラ、レンダー、合成器）
├── rc3d-editor/     — エディタライブラリ（キーマップ、コマンド、適用、Fluent UI）
├── rc3d-examples/   — デモアプリケーション（51サンプル）
├── rc3d-studio/     — デスクトップエディタホスト（ワークスペース、i18n、ケースライブラリ）
└── rc3d-cli-editor/ — ターミナルベースのエディタ
```

## レンダリング

クラスタ遅延 PBR レンダラー、完全な HDR 後処理チェーン付き。

| 機能 | 説明 |
|------|------|
| **PBR** | メタリック・ラフネス（GGX/Smith）、IBL（HDR 環境マップ、BRDF LUT） |
| **シャドウ** | CSM（4 カスケード、8% ブレンドゾーン）、全方位ポイントライトシャドウ |
| **ライティング** | クラスタベース前方（16×8×24 グリッド）、LightSetTable 重複排除（1280B→4B/描画） |
| **後処理** | TAA（YCoCg）、SSR（HIZ アクセラレーション）、SSAO、モーションブラー、DoF、ブルーム、カラーグレーディング、ボリューメトリックフォグ、自動露出 |
| **GPU カリング** | デュアルパス：CPU BVH + GPU 計算（視錐台 + HZB オクルージョン）、メッシュレットクラスタツリー |
| **選択** | スクリーンスペースアウトライン、エッジオーバーレイ、バウンディングボックス、X 線モード |
| **表示** | シェーディング、ワイヤーフレーム、非表示線、フラット、エッジ付きシェーディング |
| **適応** | 5 レベル品質コントローラ、EMA+ヒステリシス、インタラクション認識の低減 |
| **CAD ティア** | Visualization / IndustrialDisplay / ProductRendering、GPU クランピング、オービットダウングレード + クールダウン回復 |
| **合成器** | ノードベース合成グラフ（Mix ブレンドモード、Math、変換、CAD プリセット）、GPU ピンポンパスで実行 |

### PBR とマテリアル

![PBR シェーダバリアント](assets/pbr_shader_variant_viewer.png)

メタリック・ラフネスシェーディング + IBL、さらにランタイム `PbrVariantCache` がシーンフィーチャーマスクごとに最大16の専用シェーダバリアント（クリアコート、シアン、イリデッセンス、トランスミッション、異方性）をコンパイル・キャッシュします。

### レンダリング機能

![レンダリング機能](assets/render_features.png)

### ライティングと反射

![エリアライト](assets/area_light.png)
![反射](assets/reflection.png)
![シャドウ（CSM）](assets/shadow_demo.png)

エリアライト、平面反射、カスケードシャドウマップを備えたクラスタ前方ライティング。

### 後処理

![後処理](assets/post_effects.png)
![ボリューメトリックフォグ](assets/volumetric_demo.png)

完全な HDR 後処理チェーン：SSAO → SSR → DoF → ブルーム → TAA → トーンマップ、さらにレイマーチングによるボリューメトリックフォグ。

### インポートとピッキング

![インポートビューア](assets/import_viewer.png)
![ピッキング](assets/picking.png)

STL / OBJ / glTF / FBX / Inventor シーンをインポートし、面をレイピックして選択・測定・注釈付けを行います。

### 選択と表示モード

![選択アウトライン](assets/selection_outline.png)
![選択セット](assets/selection_set.png)
![インデックス付き線分/非表示線](assets/indexedlineset.png)

選択用のスクリーンスペースアウトライン、非表示線/ワイヤーフレーム表示モード、エンジニアリングビュー用のインデックス付き線分。

## シーングラフ（最小例）

```rust
use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Cube", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(root, NodeData::PerspectiveCamera(
            PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y,
                std::f32::consts::FRAC_PI_4, 800.0 / 600.0,
            ),
        ));
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE, intensity: 1.0, light_group: None,
        }));
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4, metallic: 0.0, ..Default::default()
        }));
        graph.add_child(root, NodeData::Cube(CubeNode::default()));
    });
}
```

![シーングラフ](assets/scene_graph.png)
![爆破解体ビュー](assets/exploded_view.png)

## サンプル（51 デモ）

| カテゴリ | サンプル |
|------|----------|
| はじめに | `triangle`, `cube`, `rotating_cube`, `hello_scene` |
| シーン | `scene_graph`, `annotation`, `billboard`, `environment_node`, `exploded_view`, `scripted_scene` |
| レンダリング | `pbr_scene`, `pbr_materials`, `pbr_variant_viewer`, `render_features`, `render_effects`, `material_variants`, `instancing`, `wboit_demo` |
| ライティング | `area_light`, `light_linking`, `shadow_demo`, `reflection` |
| カメラ | `stereo_camera`, `walk_camera` |
| インポート | `import_viewer` |
| アニメーション | `animation_demo`, `animation_control_panel`, `blend_animation` |
| エディタ | `selection_set`, `picking`, `markup_dimensions`, `annotation_edit` |
| エンジン | `engines_demo`, `scripted_scene` |
| エフェクト | `post_effects`, `volumetric_demo`, `decal_viewer`, `text3d` |
| 専門 | `nurbs_viewer`, `profile_viewer`, `section_caps`, `gdt_demo`, `stl_diagnostic` |
| 診断 | `adaptive_stress_test`, `large_scene_stress`, `bench` |

### CAD とエンジニアリング

![NURBS](assets/nurbs_viewer.png)
![断面キャップ](assets/section_caps.png)
![プロファイルビューア](assets/profile_viewer.png)
![GD&T / PMI](assets/gdt_pmi.png)

NURBS 曲面、断面キャップ/ハッチ、および `SceneGraph::bind_pmi` で名前付きパーツにバインドされる GD&T/PMI 注釈。

![寸法注釈](assets/markup_dimensions.png)

寸法 / 角度 / 半径 / 引出線の注釈が 3D に投影され、平面接線テキストパス（`Text2`/`Text3`）でレンダリングされます。

### 点群とインスタンス化

![点群](assets/point_cloud.png)
![インスタンス化](assets/instancing.png)

外部ストレージオクトリー点群レンダリングと、反復ジオメトリ（BatchedMesh / InstancedMesh）の GPU インスタンス化。

## ノード型（62 バリアント）

| カテゴリ | バリアント |
|------|----------|
| **グループ化** | Separator, Group, Billboard, Transform, Rotation, RotationXYZ, Coordinate3, TextureCoordinate2, Normal, ShapeHints, MaterialBinding, ResetTransform, Texture2Transform, File |
| **形状** | Triangle, Cube, Sphere, Cone, Cylinder, IndexedFaceSet, IndexedLineSet, SkinnedMesh, MorphTarget, Sprite, BatchedMesh, InstancedMesh |
| **カメラ** | PerspectiveCamera, OrthographicCamera, StereoCamera, CubeCamera |
| **ライト** | DirectionalLight, PointLight, SpotLight, AreaLight, HemisphereLight, LightProbe |
| **トラバーサル** | Lod, Switch, MultipleCopy, SectionPlane, PickStyle, EventCallback |
| **注釈** | Text2, Text3, Measurement, Markup, Annotation, Font |
| **マニピュレータ** | TransformManip, Dragger, Rotation |
| **専門** | ExplodedView, ReflectionPlane, Decal, RayTracing, Volume, PointCloud, Environment, Material |
| **拡張性** | HandlerNode(Arc\<dyn NodeHandler\>), Custom(u16, Box\<dyn CustomNodeData\>) |

## 性能特性

| シーン | オブジェクト | 描画呼び出し | パイプライン | フレーム時間 |
|------|---------|------------|------|--------|
| ストレステスト | 10K | ~10K | インスタンスバッチ | ~5ms CPU |
| ストレステスト（静的） | 10K | ~10K | 静的フレーム高速パス | ~1ms CPU |
| インポートビューア | 1-100K | 変動 | ストリーミングメッシュ | ~8-16ms |
| ターゲット（GPU） | 1M+ | 間接 | GPU ドリブン | TBD |

主要な最適化：
- **ライト重複排除**：描画呼び出しごとに 1280B → 4B（LightSetTable）
- **静的フレーム高速パス**：シーンがアイドル時はカリング作業ゼロ
- **フレーム割り当て再利用**：8 つの Vec をフレーム間で再利用（@1M オブジェクトで約 60MB 節約）
- **BVH インクリメンタル**：ダーティ AABB のみが BVH 更新をトリガー
- **ダイレクトキャッシュ出力**：FlatDrawCache をトラバーサル中に埋める（変換パスなし）
- **プール拡張**：phong 64K、flat 32K、メッシュキャッシュ 4K

## 開発

```bash
cargo check --workspace          # 高速コンパイルチェック
cargo test                       # 328 テスト
cargo build -p rc3d-examples --examples
cargo run -p rc3d-studio         # デスクトップエディタ
cargo clippy --workspace         # lint チェック
```

## Studio デスクトップエディタ

![Studio UI](assets/studio-ui.png)

`rc3d-studio` はフラッグシップのデスクトップアプリケーション：

- **ワークスペース**：Model / LookDev / Compositor のクイックレイアウト
- **ドック**：サイドドック（Hierarchy+Inspector 分割、Render、History、Assets）、ボトムドック（Document、Compositor）、移動可能なツールストリップ
- **合成器エディタ**：Blender スタイルのノードグラフ（`egui-snarl`）、2 レベルの Add メニュー、折りたたみ状態の永続化
- **ケースライブラリ**：24 のパラメータ/プロセスデモケース、ステップバイステップのガイダンス付き
- **i18n**：英語 / 簡体字中国語（450 キーカタログ）
- **キーマップ**：デフォルトショートカット、`%APPDATA%\rustcoin3d\ui-prefs.json` にユーザーごとの上書きを保存
- **CAD マトリックスチェック**：`rc3d-studio --cad-matrix` でティア/合成器検証マトリックスを実行

## 依存関係

| crate | 目的 |
|-------|------|
| wgpu 30 | GPU 抽象化（Vulkan/Metal/DX12） |
| winit 0.30 | ウィンドウ作成とイベントループ |
| egui 0.36 / eframe 0.36 | イミディエイトモード UI（エディタ + Studio） |
| glam 0.29 | 線形代数（Vec3、Mat4、Quat） |
| slotmap | シーングラフの安定 ID アリーナストレージ |
| glyphon 0.12 | GPU テキストレンダリング（HUD） |
| rayon | 並列トラバーサル |
| rhai | 埋め込みスクリプティング |
| meshopt | メッシュ最適化（メッシュレット、LOD） |
| serde/serde_json | シリアライゼーション |
| image | テクスチャ読み込み |
| tracy-client | GPU/CPU プロファイリング |

## ライセンス

BSD-3-Clause
