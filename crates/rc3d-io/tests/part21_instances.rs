//! Part21 instance and parameter parsing tests.

use rc3d_io::step::model::ComplexMapping;
use rc3d_io::step::part21::instance::parse_instance;
use rc3d_io::step::part21::params::parse_param_list;
use rc3d_io::step::value::StepValue;

#[test]
fn parse_params_nested() {
    let (val, rest) =
        parse_param_list("#2, (3.0, 4.0), LENGTH_MEASURE(0.001), $").unwrap();
    assert!(rest.trim_start().is_empty());
    let list = val.as_list().unwrap();
    assert_eq!(list.len(), 4);
    assert!(matches!(list[3], StepValue::Omitted));
}

#[test]
fn parse_simple_cartesian_point() {
    let (inst, rest) = parse_instance("#1 = CARTESIAN_POINT('', (0.0, 1.0, 2.0));").unwrap();
    assert_eq!(inst.id, 1);
    assert_eq!(inst.records.len(), 1);
    assert_eq!(inst.records[0].keyword, "CARTESIAN_POINT");
    assert!(rest.trim().is_empty());
}

#[test]
fn parse_subsuper_inside_preserves_records() {
    let input = "#1 = (\n\
BOUNDED_CURVE()\n\
B_SPLINE_CURVE(2,(#10,#11),.UNSPECIFIED.,.F.,.F.)\n\
B_SPLINE_CURVE_WITH_KNOTS((3,2,3),(0.625,0.667,0.75),.UNSPECIFIED.)\n\
CURVE()\n\
);\n";
    let (inst, _) = parse_instance(input).unwrap();
    assert_eq!(inst.mapping, ComplexMapping::Internal { leaf_index: 3 });
    assert_eq!(inst.records.len(), 4);
    let knots = inst
        .records
        .iter()
        .find(|r| r.keyword == "B_SPLINE_CURVE_WITH_KNOTS")
        .expect("knots record");
    assert!(knots.params.as_list().unwrap().len() >= 2);
}

#[test]
fn parse_subsuper_format1_external() {
    let input = "#1 = (LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.MILLI.,.METRE.))ENTITY(1.0, 2.0);";
    let (inst, rest) = parse_instance(input).unwrap();
    assert_eq!(inst.mapping, ComplexMapping::Internal { leaf_index: inst.records.len().saturating_sub(1) });
    assert!(inst.records.len() >= 3);
    assert!(rest.trim().is_empty() || rest.trim().starts_with('#'));
}
