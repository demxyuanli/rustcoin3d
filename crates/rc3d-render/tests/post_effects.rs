use rc3d_render::PostEffectSettings;

#[test]
fn post_effect_defaults_exist() {
    let settings = PostEffectSettings::default();
    let _ = settings;
}
