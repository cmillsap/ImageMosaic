; Inno Setup script for the Image Mosaic Generator.
;
; Wraps the PyInstaller one-folder build in dist\ImageMosaic into a single
; setup executable. Build the app first:
;
;   venv\Scripts\pyinstaller.exe packaging\ImageMosaic.spec --noconfirm
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; packaging\build.ps1 does both.

#define AppName "Image Mosaic Generator"
#define AppVersion "1.3.1"
#define AppPublisher "Chris Millsap"
#define AppExeName "ImageMosaic.exe"
#define SourceDir "..\dist\ImageMosaic"

[Setup]
; Never reuse this GUID in another product; it is how Windows recognises an
; existing install and offers to upgrade it in place.
AppId={{67F27EDD-B156-4BE9-8CD1-2973148D845C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\ImageMosaic
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

; Per-user by default, so the common case installs with no UAC prompt at all.
; A user who wants it for everyone can still choose that on the first page.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

; The payload is ~350 MB of Qt, OpenCV and SciPy; solid LZMA2 takes it to
; roughly a third of that, at the cost of a slower compile.
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

WizardStyle=modern
SetupIconFile=icon.ico
OutputDir=..\dist
OutputBaseFilename=ImageMosaic-{#AppVersion}-Setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}"; DestName: "README.md"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Analysed tiles are cached under %TEMP% and a large library runs to
; gigabytes, so uninstalling should take them with it.
Type: filesandordirs; Name: "{%TEMP}\image_mosaic_tile_cache"

[Code]
function InitializeSetup(): Boolean;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  // PyQt6 6.10 and the OpenCV wheels are built against the Windows 10 SDK;
  // older releases fail at DLL load with an error the user cannot act on.
  if (Version.Major < 10) then
  begin
    MsgBox('{#AppName} requires Windows 10 or later.', mbCriticalError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
