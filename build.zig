const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const native_dir = b.option([]const u8, "native-lib-dir", "Directory with gyro_native.lib (Windows only)");
    if (native_dir != null and target.result.os.tag != .windows)
        @panic("The native backend requires a Windows target.");

    const options = b.addOptions();
    options.addOption(bool, "native", native_dir != null);
    const core = b.addModule("gyro_dx", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const exe = b.addExecutable(.{
        .name = "gyro-dx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "gyro_dx", .module = core }},
        }),
    });
    exe.root_module.addOptions("build_options", options);
    if (native_dir) |dir| {
        // The Agility SDK finds D3D12SDKVersion and D3D12SDKPath in the executable's export table.
        exe.rdynamic = true;
        exe.root_module.addLibraryPath(.{ .cwd_relative = dir });
        exe.root_module.linkSystemLibrary("gyro_native", .{});
    }
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step("run", "Run the command line tool").dependOn(&run.step);

    const test_step = b.step("test", "Run parser, decoder, cache, and conformance tests");
    const tests = b.addTest(.{ .root_module = core });
    test_step.dependOn(&b.addRunArtifact(tests).step);
    const gyro_module = b.createModule(.{
        .root_source_file = b.path("src/gyro.zig"), .target = target, .optimize = optimize,
    });
    const vector_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/gyro_vectors.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gyro", .module = gyro_module }},
    }) });
    test_step.dependOn(&b.addRunArtifact(vector_tests).step);
}
