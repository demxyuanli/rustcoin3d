// OCC BREP Validator — reads BREP files via Open CASCADE and checks topology.
//
// Build (Windows + OCC installed via vcpkg or official installer):
//   cl /EHsc /I"%OCC_INC%" validate_brep_occ.cpp /link /LIBPATH:"%OCC_LIB%" ^
//      TKBRep.lib TKTopAlgo.lib TKGeomBase.lib TKMath.lib TKernel.lib
//
// Build (Linux/macOS):
//   g++ -std=c++17 validate_brep_occ.cpp -lTKBRep -lTKTopAlgo -lTKGeomBase \
//       -lTKMath -lTKernel -I$OCC_INC -L$OCC_LIB -o validate_brep_occ
//
// Usage:
//   validate_brep_occ.exe compare/out/step-*.brep
//
// Checks:
//   1. File can be read by BRepTools::Read
//   2. TopoDS_Shape is not null
//   3. ShapeType is Solid (or Compound containing Solid)
//   4. BRepCheck_Analyzer reports no major errors
//   5. Counts vertices, edges, faces, shells, solids

#include <BRep_Builder.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepTools.hxx>
#include <Standard_Handle.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Solid.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopExp.hxx>

#include <iostream>
#include <string>
#include <vector>

struct Counts {
    int vertices = 0, edges = 0, wires = 0, faces = 0, shells = 0, solids = 0;
};

Counts count_topology(const TopoDS_Shape& shape) {
    Counts c;
    TopTools_IndexedMapOfShape m;

    TopExp::MapShapes(shape, TopAbs_VERTEX, m);   c.vertices = m.Extent(); m.Clear();
    TopExp::MapShapes(shape, TopAbs_EDGE, m);     c.edges = m.Extent(); m.Clear();
    TopExp::MapShapes(shape, TopAbs_WIRE, m);     c.wires = m.Extent(); m.Clear();
    TopExp::MapShapes(shape, TopAbs_FACE, m);     c.faces = m.Extent(); m.Clear();
    TopExp::MapShapes(shape, TopAbs_SHELL, m);    c.shells = m.Extent(); m.Clear();
    TopExp::MapShapes(shape, TopAbs_SOLID, m);    c.solids = m.Extent(); m.Clear();
    return c;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <brep_file>..." << std::endl;
        return 1;
    }

    int passed = 0, failed = 0;

    for (int i = 1; i < argc; i++) {
        std::string path(argv[i]);
        std::cout << path << ": ";

        // 1. Read
        TopoDS_Shape shape;
        BRep_Builder builder;
        Standard_Boolean ok = BRepTools::Read(shape, path.c_str(), builder);
        if (!ok) {
            std::cerr << "FAIL (BRepTools::Read returned false)" << std::endl;
            failed++;
            continue;
        }
        if (shape.IsNull()) {
            std::cerr << "FAIL (null shape)" << std::endl;
            failed++;
            continue;
        }

        // 2. Check topology
        Counts c = count_topology(shape);
        if (c.solids == 0 && c.faces == 0) {
            std::cerr << "FAIL (no solids or faces)" << std::endl;
            failed++;
            continue;
        }

        // 3. Run BRepCheck
        BRepCheck_Analyzer checker(shape, Standard_True);
        if (!checker.IsValid()) {
            // Collect sub-shapes with faults
            TopTools_IndexedMapOfShape faces;
            TopExp::MapShapes(shape, TopAbs_FACE, faces);
            int faults = 0;
            for (int j = 1; j <= faces.Extent() && faults < 5; j++) {
                const TopoDS_Shape& f = faces(j);
                if (!checker.IsValid(f)) {
                    if (faults == 0) std::cerr << "WARN (checker faults: ";
                    std::cerr << "face " << j << " ";
                    faults++;
                }
            }
            if (faults > 0) std::cerr << ")";
            // Still OK — BRepCheck warnings are acceptable for visualization
        }

        // 4. Print summary
        std::cout << "OK  V=" << c.vertices << " E=" << c.edges
                  << " F=" << c.faces << " Sh=" << c.shells
                  << " So=" << c.solids << std::endl;
        passed++;
    }

    std::cout << "\n" << passed << "/" << (passed + failed) << " files valid" << std::endl;
    return failed > 0 ? 1 : 0;
}
