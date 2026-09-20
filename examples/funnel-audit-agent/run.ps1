# PowerShell runner for the funnel audit demo.
Write-Output "Dry run (no paid call):"
bruv demo funnel-audit --backend simple-jev
Write-Output ""
Write-Output "Execute the real evaluation:"
bruv demo funnel-audit --backend simple-jev --execute