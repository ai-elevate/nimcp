#!/usr/bin/env node
/**
 * NIMCP Metrics Example for Node.js
 * Demonstrates metrics collection for Tableau and PowerBI
 */

const nimcp = require('./index');

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    try {
        console.log('NIMCP Metrics Example\n');

        // Create metrics collector
        const metrics = new nimcp.MetricsCollector({
            directory: './nodejs_metrics'
        });
        console.log('Metrics collector created\n');

        // Record various metrics
        console.log('Recording metrics...');

        // Counter metrics
        metrics.recordCounter('forward_passes', 1000, nimcp.METRIC_CATEGORY_PERFORMANCE);
        metrics.recordCounter('training_examples', 5000, nimcp.METRIC_CATEGORY_LEARNING);

        // Gauge metrics
        metrics.recordGauge('learning_rate', 0.001, nimcp.METRIC_CATEGORY_LEARNING);
        metrics.recordGauge('accuracy', 0.95, nimcp.METRIC_CATEGORY_INFERENCE);
        metrics.recordGauge('memory_usage_mb', 256.5, nimcp.METRIC_CATEGORY_MEMORY);

        // Timer metrics
        metrics.recordTimer('forward_time_ms', 15.5, nimcp.METRIC_CATEGORY_PERFORMANCE);
        metrics.recordTimer('backward_time_ms', 22.3, nimcp.METRIC_CATEGORY_PERFORMANCE);

        // Timing example
        const startTime = Date.now();
        await sleep(100);
        const duration = Date.now() - startTime;
        metrics.recordTimer('data_processing', duration, nimcp.METRIC_CATEGORY_PERFORMANCE);
        console.log('Processed data in ' + duration + 'ms');

        // Simulate training loop with metrics
        console.log('\nSimulating training loop with metrics...');
        for (let epoch = 0; epoch < 5; epoch++) {
            const epochStart = Date.now();

            // Record epoch-specific metrics
            metrics.recordGauge('current_epoch', epoch, nimcp.METRIC_CATEGORY_LEARNING);
            metrics.recordGauge('loss', 0.5 / (epoch + 1), nimcp.METRIC_CATEGORY_LEARNING);

            // Simulate training time
            await sleep(50);

            const epochDuration = Date.now() - epochStart;
            metrics.recordTimer('epoch_' + epoch, epochDuration, nimcp.METRIC_CATEGORY_LEARNING);
        }

        // Get and display statistics
        console.log('\nMetrics Statistics:');
        const stats = JSON.parse(metrics.getStats());
        console.log('  Total metrics recorded:', stats.total_recorded);
        console.log('  Total metrics flushed:', stats.total_flushed);
        console.log('  Buffer count:', stats.buffer_count);

        // Flush metrics to disk
        console.log('\nFlushing metrics...');
        const count = metrics.flush();
        console.log('  Flushed ' + count + ' metrics to disk');

        // Export for Tableau
        const tableauFile = './nodejs_metrics_tableau.csv';
        console.log('\nExporting to Tableau CSV: ' + tableauFile);
        metrics.exportTableauCsv(tableauFile);

        // Export for PowerBI
        const powerbiFile = './nodejs_metrics_powerbi.json';
        console.log('Exporting to PowerBI JSON: ' + powerbiFile);
        metrics.exportPowerBiJson(powerbiFile);

        console.log('\nExample completed successfully!');

    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main();
